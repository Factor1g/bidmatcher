"""
Benchmark pipeline
==================
Lépések:
 
  1. Teszthalmaz generálása (egyszer):
     python -m bidmatch.ml.benchmark create --output data/testset.csv --samples 200
 
  2. Kézi annotálás:
     Nyisd meg a data/testset.csv-t Excelben.
     Töltsd ki a 'correct_match' oszlopot minden sorhoz:
       1     = a pre_suggested helyes egyezés
       0     = rossz
       0.5   = részleges (opcionális)
       üres  = kihagyandó sor
 
  3. Benchmark futtatása:
     python -m bidmatch.ml.benchmark run --testset data/testset.csv --top-k 5
"""

 
import argparse
from pathlib import Path
from collections import defaultdict
 
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
 
from ..ml.bm25 import BM25
from ..ml.tokenizer import tokenize
from ..ml.rrf import reciprocal_rank_fusion
from ..ml.hybrid_retriever import HybridRetriever as _HybridRetriever
from ..config.setting import (
    TRAINING_DATA_PATH, EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL,
)

class BM25Retriever:
    def __init__(self, corpus_df: pd.DataFrame, k1: float = 1.5, b: float = 0.6):
        self.corpus_df = corpus_df.reset_index(drop=True)
        corpus_texts = (
            corpus_df["item_description"].astype(str).str.strip()
            + " | "
            + corpus_df["unit"].astype(str).str.strip()
        ).tolist()
        print(f"Building BM25 index for {len(corpus_texts)} corpus items...")
        tokenized_corpus = [tokenize(text) for text in corpus_texts]
        self.bm25 = BM25(k1=k1, b=b).fit(tokenized_corpus)
        print(f"BM25 index built successfully for {len(corpus_texts)} items.")

        def search(query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
            query_text = f"{query.strip()} | {unit.strip()}" if unit.strip() else query.strip()
            scores = self.bm25.get_score(tokenize(query_text))
            top_idx = np.argsort(scores)[::-1][:top_k]
            result = self.corpus_df.iloc[top_idx].copy()
            result["bm25_score"] = scores[top_idx]
            return result.reset_index(drop=True)

class BiEncoderOnlyRetriever:
    def __init__(self, corpus_df: pd.DataFrame, embeddings: np.ndarray, model_name: str = EMBEDDING_MODEL_NAME):
        self.df = corpus_df.reset_index(drop=True)
        self.embeddings = embeddings
        print(f"Loading Bi-Encoder model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
        query_text = f"{query.strip()} | {unit.strip()}" if unit.strip() else query.strip()
        q_emb = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        result = self.df.iloc[top_idx].copy()
        result["bi_score"] = scores[top_idx]
        return result.reset_index(drop=True)

class HybridRRFRetriever():
    def __init__(self, corpus_df: pd.DataFrame, embeddings: np.ndarray):
        self._retriever = _HybridRetriever(cross_encoder_model_name= None)
        self._retriever.build_index(corpus_df, embeddings)

    def search(self, query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
        result = self._retriever.search(query, unit, top_k=top_k, use_cross_encoder=False)
        return result if result is not None else pd.DataFrame()

class HíbridFullRetriever:
    def __init__(self, corpus_df: pd.DataFrame, embeddings: np.ndarray):
        self._retriever = _HybridRetriever()
        self._retriever.build_index(corpus_df, embeddings)

    def search(self, query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
        result = self._retriever.search(query, unit, top_k=top_k)
        return result if result is not None else pd.DataFrame()


        


 