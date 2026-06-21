from typing import Optional
import pandas as pd
import numpy as np
from .tokenizer import tokenize
from .bm25 import BM25
from sklearn.metrics.pairwise import cosine_similarity
from .rrf import reciprocal_rank_fusion

from ..config.setting import EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL
"""
    Kétlépéses hibrid retrieval:
      1. BM25 + Bi-Encoder párhuzamosan → top-N jelöltek
      2. RRF fusion → összesített ranglista
      3. (opcionális) Cross-Encoder re-ranking → top-K végeredmény
 
    Használat:
        retriever = HybridRetriever()
        retriever.build_index(df)            # egyszer, korpusz betöltésekor
        results = retriever.search("DN50 gömbcsap MOFÉM")
"""
class HybridRetriever:
    def __init__(
            self,
            bi_encoder_model_name: str = EMBEDDING_MODEL_NAME,
            cross_encoder_model_name: str = CROSS_ENCODER_MODEL,
            bm25_k1: float = 1.5,
            bm25_b: float = 0.6,
            bm25_weight: float = 0.4,
            bi_encoder_weight: float = 0.6,
            rrf_k: int = 60,
            top_n: int = 50,
            top_k: int = 5,
            uncertainty_threshold: float = 0.0
    ):
        self.bi_encoder_model_name = bi_encoder_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_weight = bm25_weight
        self.bi_encoder_weight = bi_encoder_weight
        self.rrf_k = rrf_k
        self.top_n = top_n
        self.top_k = top_k
        self.uncertainty_threshold = uncertainty_threshold

        #Initialized in build_index
        self.df: pd.DataFrame = None
        self.bm25: BM25 = None
        self.corpus_embeddings: np.ndarray = None
        self.bi_encoder = None
        self.cross_encoder = None
        self.corpus_texts: list[str] = []

    def build_index(self,
                    df: pd.DataFrame,
                    embeddings: np.ndarray,
                    desc_col: str = "item_description",
                    unit_col: str= "unit"
                    ) -> 'HybridRetriever':
        """
            Felépíti a BM25 és bi-encoder indexet a korpuszból. 
            Args:
                df         : korpusz DataFrame
                embeddings : előre generált embeddings (ha None, generálja)
                desc_col   : leírás oszlop neve
                unit_col   : mértékegység oszlop neve
        """
        self.df = df.reset_index(drop=True)

        self.corpus_texts = (df[desc_col].astype(str).str.strip() + 
                            " | " + df[unit_col].astype(str).str.strip()
                            ).tolist()
        
        print("[BM25] Tokenizálás és index építés...")
        tokenized = [tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25(k1=self.bm25_k1, b=self.bm25_b).fit(tokenized)
        print(f"[BM25] Index kész: {len(tokenized)} dokumentum")

        if embeddings is not None:
            self.corpus_embeddings = embeddings
            print(f"[Bi-Encoder] Előre generált embeddings betöltve: {embeddings.shape}")
        else:
            print("[Bi-Encoder] Embeddings generálása...")
            from sentence_transformers import SentenceTransformer
            self.bi_encoder = SentenceTransformer(self.bi_encoder_model_name)
            self.corpus_embeddings = self.bi_encoder.encode(
                self.corpus_texts,
                show_progress_bar = True,
                convert_to_numpy = True,
                normalize_embeddings = True,
                batch_size = 32
            )     
            print(f"[Bi-Encoder] Embeddings kész: {self.corpus_embeddings.shape}")
        return self

    def _get_bi_encoder(self):
        if self.bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self.bi_encoder = SentenceTransformer(self.bi_encoder_model_name)
        return self.bi_encoder
    
    def _get_cross_encoder(self):
        if self.cross_encoder is None and self.cross_encoder_model_name:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        return self.cross_encoder

    def search(self, query: str, unit: str = "", top_k: int= None, use_cross_encoder: bool = True) -> pd.DataFrame:
        """
            Hibrid keresés: BM25 + Bi-Encoder → RRF → (Cross-Encoder).
    
            Args:
                query           : keresési szöveg
                unit            : mértékegység (opcionális, javítja a pontosságot)
                top_k           : végső találatok száma (None = self.top_k)
                use_cross_encoder: Cross-Encoder re-ranking bekapcsolása
    
            Returns:
                DataFrame a top találatokkal, vagy None ha UNCERTAIN
        """
        assert self.df is not None, "Index nincs felépítve. Hívja meg előbb build_index()-et."

        top_k = top_k or self.top_k
        query_text = f"{query.strip()} | {unit.strip()}" if unit.strip() else query.strip()

        #BM25 scoring
        bm25_scores = self.bm25.get_score(tokenize(query_text))
        bm25_ranking = list(np.argsort(bm25_scores)[::-1][:self.top_n])

        #Bi-Encoder scoring
        bi_encoder = self._get_bi_encoder()
        q_emb = bi_encoder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        bi_scores = cosine_similarity(q_emb, self.corpus_embeddings)[0]
        bi_ranking = list(np.argsort(bi_scores)[::-1][:self.top_n])

        #RRF fusion
        fused = reciprocal_rank_fusion(
            rankings=[bm25_ranking, bi_ranking],
            weights=[self.bm25_weight, self.bi_encoder_weight],
            k=self.rrf_k
        )

        # UNCERTAIN ellenőrzés
        if fused and fused[0][1] < self.uncertainty_threshold:
            return None

        #Top-N jelöltek kiválasztása ffr alapján
        top_candidates_idx = [idx for idx, _ in fused[:self.top_n]]
        top_rrf_scores = {idx: score for idx, score in fused[:self.top_n]}

        candidates = self.df.iloc[top_candidates_idx].copy()
        candidates["bm25_score"] = bm25_scores[top_candidates_idx]
        candidates["bi_score"] = bi_scores[top_candidates_idx]
        candidates["rrf_score"] = [top_rrf_scores[i] for i in top_candidates_idx]

        #Cross-Encoder re-ranking
        if use_cross_encoder and self.cross_encoder_model_name:
            ce = self._get_cross_encoder()
            candidate_texts = (
                candidates["item_description"].astype(str).str.strip() + 
                " | " + candidates["unit"].astype(str).str.strip()
            ).tolist()

            ce_scores = ce.predict([(query_text, ct) for ct in candidate_texts])
            
            candidates["cross_encoder_score"] = ce_scores
            result = candidates.sort_values("cross_encoder_score", ascending=False)

        else:
            result = candidates.sort_values("rrf_score", ascending=False)

        return result.head(top_k).reset_index(drop=True)
