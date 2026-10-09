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

    def search(self,query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
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

class HybridFullRetriever:
    def __init__(self, corpus_df: pd.DataFrame, embeddings: np.ndarray):
        self._retriever = _HybridRetriever()
        self._retriever.build_index(corpus_df, embeddings)

    def search(self, query: str, unit: str = "", top_k: int = 5) -> pd.DataFrame:
        result = self._retriever.search(query, unit, top_k=top_k)
        return result if result is not None else pd.DataFrame()


def create_testset(corpus_path: Path = TRAINING_DATA_PATH, embeddings_path: Path = EMBEDDINGS_PATH, output_path: Path = Path("data/testset.csv"), samples: int = 200) -> pd.DataFrame:
    """
    Generate a test set for manual annotation.
 
    What it does:
      - Takes n_samples random rows from the corpus → these will be the queries
      - For each query, finds the best match according to the bi-encoder (pre_suggested)
      - Leaves the correct_match column empty → this needs to be filled manually
  
    Output CSV:
      query_description  : queried item description
      query_unit         : queried item unit
      correct_match      : ← cell for annotation (1 / 0 / 0.5 / empty)
      pre_suggested      : bi-encoder best match
      suggested_unit     : suggested unit for the pre_suggested match
      source_file        : source file of the query (if available)
    """
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_parquet(corpus_path)
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    sample = df.sample(n=min(samples, len(df)), random_state=42).copy()

    suggestions, suggested_units = [], []
    
    for _, row in sample.iterrows():
        query_text = f"{row['item_description'].strip()} | {row['unit'].strip()}"
        q_emb = SentenceTransformer(EMBEDDING_MODEL_NAME).encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb,embeddings)[0]

        #Best match that is not the same as the query itself
        for idx in np.argsort(sims)[::-1]:
            candidate = df.iloc[idx]
            if candidate['item_description'] != row['item_description']:
                suggestions.append(candidate['item_description'])
                suggested_units.append([candidate['unit']])
                break
        else:
            suggestions.append("")
            suggested_units.append("")

    testset = pd.DataFrame({
        "query_description": sample['item_description'].values,
        "query_unit": sample['unit'].values,
        "correct_match": "",
        "pre_suggested": suggestions,
        "suggested_unit": suggested_units,
        "source_file": sample.get("source_file", pd.Series("")).values
    })  

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    testset.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Teszthalmaz mentve: {output_path}")
    print(f"  {len(testset)} sor — töltsd ki a 'correct_match' oszlopot:")
    print("    1     = pre_suggested helyes egyezés")
    print("    0     = rossz egyezés")
    print("    0.5   = részleges egyezés")
    print("    üres  = kihagyandó sor")
 
    return testset  

def _similarity_score(correct: str, candidate: str, threshold: float = 0.8) -> float:
    """
    Overlap based similarity score between two strings.
    Returns 1.0 if the overlap is above the threshold, 0.0 otherwise
    """
    correct_tokens   = set(correct.lower().split())
    candidate_tokens = set(candidate.lower().split())
    if not correct_tokens or not candidate_tokens:
        return 0.0
    overlap = len(correct_tokens & candidate_tokens)
    shorter = min(len(correct_tokens), len(candidate_tokens))
    score = overlap / shorter
    return 1.0 if score >= threshold else 0.0

def _hits(results: pd.DataFrame, correct: str, top_k: int) -> list[float]:
    """
    Returns a top_k long list:
    1.0 if the correct string is found in the results, 0.0 otherwise.
    """
    hit_list = [_similarity_score(correct, str(row['item_description']).strip()) for _, row in results.iterrows()]

    #Padding if results are less than top_k
    hit_list += [0.0] * (top_k - len(hit_list))
    return hit_list[:top_k]

def precision_at_k(hits: list[float], top_k: int) -> float:
    """
    Precision@k = (# of relevant items in top k) / k
    """
    return sum(1 for hit in hits[:top_k] if hit > 0) / top_k

def mrr_at_k(hits:list[float], top_k: int) -> float:
    """
    Mean Reciprocal Rank (MRR) at k
    MRR = 1 / rank of the first relevant item
    """
    for rank, hit in enumerate(hits[:top_k], start=1):
        if hit > 0:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(hits: list[float], top_k:int):
    """
    Normalized Discounted Cumulative Gain (NDCG) at k
    NDCG = DCG / IDCG
    DCG = sum((2^rel - 1) / log2(rank + 1)) for relevant items
    IDCG = ideal DCG (max possible DCG)
    """
    dcg = sum(hit/np.log2(i + 2) for i, hit in enumerate(hits[:top_k]))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(sum(hit > 0 for hit in hits), top_k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(retriver, testset: pd.DataFrame, top_k: int = 5) -> dict:
    """
    Evaluate a retriever on a testset.
    
    Returns: dict {P@1, P@K, MRR@K, NDCG@K, n_queries}
    """
    p1, pk, mrr, ndcg = [], [], [], []

    for _, row in testset.iterrows():
        query = str(row['query_description'].strip().lower())
        unit = str(row['query_unit'].strip().lower())
        correct = str(row['correct_match'].strip().lower())

        if not correct or correct in ["nan", "none", ""]:
            continue

        # try:
        #     match = str(correct)

        #     if match == 0:
        #         continue 



        # except ValueError:
        #     print(f"Warning: Invalid correct_match value '{correct}' for query '{query}'. Skipping.")
        #     continue

        results = retriver.search(query, unit)

        if results.empty:
            p1.append(0.0)
            pk.append(0.0)
            mrr.append(0.0)
            ndcg.append(0.0)

            continue

        hits = _hits(results, correct, top_k)
        p1.append(precision_at_k(hits, 1))
        pk.append(precision_at_k(hits, top_k))
        mrr.append(mrr_at_k(hits, top_k))
        ndcg.append(ndcg_at_k(hits, top_k))

    if not p1:
        print("No valid queries found in the testset. Evaluation aborted.")
        return {}

    return {
        "P@1":          round(np.mean(p1),   3),
        f"P@{top_k}":   round(np.mean(pk),   3),
        f"MRR@{top_k}": round(np.mean(mrr),  3),
        f"NDCG@{top_k}":round(np.mean(ndcg), 3),
        "n_queries":    len(p1),
    }

def run_full_benchmark(testset_path: Path, output_path: Path = Path("data/benchmark_results.csv"), corpus_path: Path = TRAINING_DATA_PATH, embeddings_path: Path = EMBEDDINGS_PATH, top_k: int = 5) -> pd.DataFrame:
    """
    Run the full benchmark on the given testset.
    Returns a DataFrame with the results for each retriever.
    """
    print(f"Loading")
    corpus_df = pd.read_parquet(corpus_path)
    embeddings = np.load(embeddings_path)
    testset = pd.read_csv(testset_path, encoding='cp1250', sep=None, engine='python')

    testset = testset[
        testset["correct_match"].notna() &
        (testset["correct_match"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    print(f"Annotált tesztsorok: {len(testset)}")
 
    systems = {
        "1. BM25 only":       BM25Retriever(corpus_df),
        "2. Bi-Encoder only": BiEncoderOnlyRetriever(corpus_df, embeddings),
        "3. Hybrid (RRF)":    HybridRRFRetriever(corpus_df, embeddings),
        "4. Hybrid + CE":     HybridFullRetriever(corpus_df, embeddings),
    }
 
    results = {}
    for name, retriever in systems.items():
        print(f"\n── {name} ──")
        results[name] = evaluate(retriever, testset)
 
    result_df = pd.DataFrame(results).T
 
    print("\n" + "═" * 60)
    print("BENCHMARK EREDMÉNYEK")
    print("═" * 60)
    print(result_df.to_string())
    print("═" * 60)
 
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, encoding="utf-8-sig")
    print(f"\n✓ Eredmények mentve: {output_path}")
 
    return result_df

def create_test_set(
    corpus_path: Path = TRAINING_DATA_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    output_path: Path = Path("data/testset.csv"),
    n_samples: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generál egy annotálásra kész teszthalmazt a korpuszból — B megoldás.

    Minden query mellé három rendszer tippjét generálja egymás mellett,
    így az annotáláskor látod az összes rendszer javaslatát, és nem vagy
    elfogult egyikük felé sem.

    Fontos: a tippek NEM jöhetnek ugyanabból a source_file-ból mint a query
    — ez megakadályozza hogy a modell saját magát találja meg.

    Output CSV oszlopok:
      query_description     : keresési szöveg
      query_unit            : mértékegység
      query_source          : forrás Excel fájl (tájékoztatásul)
      bm25_suggested        : BM25 legjobb tippje
      bm25_unit             : BM25 tipp mértékegysége
      bi_suggested          : Bi-Encoder legjobb tippje
      bi_unit               : Bi-Encoder tipp mértékegysége
      hybrid_suggested      : Hibrid (RRF) legjobb tippje
      hybrid_unit           : Hibrid tipp mértékegysége
      correct_match         : ← KÉZZEL KITÖLTENDŐ
                                  bm25 / bi / hybrid  = melyik rendszer tippje helyes
                                  none                = egyik sem helyes
                                  <szöveg>            = saját helyes leírás beírva
    """
    print(f"Korpusz betöltése: {corpus_path}")
    corpus_df  = pd.read_parquet(corpus_path)
    embeddings = np.load(embeddings_path)
    print(f"Korpusz mérete: {len(corpus_df)} sor")

    sample = corpus_df.sample(n=min(n_samples, len(corpus_df)), random_state=seed).copy()

    # Modellek betöltése
    print(f"[Bi-Encoder] Modell betöltése: {EMBEDDING_MODEL_NAME}")
    bi_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    corpus_texts = (
        corpus_df["item_description"].astype(str).str.strip()
        + " | "
        + corpus_df["unit"].astype(str).str.strip()
    ).tolist()

    print("[BM25] Index építés...")
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25(k1=1.5, b=0.6).fit(tokenized_corpus)
    print("[BM25] Kész")

    rows = []

    for _, row in sample.iterrows():
        query_desc   = str(row["item_description"]).strip()
        query_unit   = str(row.get("unit", "")).strip()
        query_source = str(row.get("source_file", "")).strip()
        query_text   = f"{query_desc} | {query_unit}" if query_unit else query_desc

        # Korpusz szűkítése: ugyanabból a source_file-ból ne jöjjön tipp
        # Ez megakadályozza hogy a modell saját magát vagy közvetlen szomszédját
        # találja meg — ami triviálisan helyes lenne és torzítaná a metrikákat
        if query_source:
            other_mask = corpus_df["source_file"].astype(str) != query_source
        else:
            # Ha nincs source_file info, legalább a pontos szövegegyezést zárjuk ki
            other_mask = corpus_df["item_description"].astype(str) != query_desc

        other_df         = corpus_df[other_mask].reset_index(drop=True)
        other_embeddings = embeddings[other_mask.values]
        other_tokenized  = [t for t, m in zip(tokenized_corpus, other_mask) if m]

        # ── BM25 tipp ────────────────────────────────────────────────────────
        bm25_other = BM25(k1=1.5, b=0.6).fit(other_tokenized)
        bm25_scores = bm25_other.get_score(tokenize(query_text))
        bm25_idx    = int(np.argsort(bm25_scores)[::-1][0])
        bm25_sugg   = other_df.iloc[bm25_idx]["item_description"]
        bm25_unit   = other_df.iloc[bm25_idx]["unit"]

        # ── Bi-Encoder tipp ──────────────────────────────────────────────────
        q_emb     = bi_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        bi_scores = cosine_similarity(q_emb, other_embeddings)[0]
        bi_idx    = int(np.argsort(bi_scores)[::-1][0])
        bi_sugg   = other_df.iloc[bi_idx]["item_description"]
        bi_unit   = other_df.iloc[bi_idx]["unit"]

        # ── Hibrid RRF tipp ──────────────────────────────────────────────────
        bm25_ranking = list(np.argsort(bm25_scores)[::-1][:50])
        bi_ranking   = list(np.argsort(bi_scores)[::-1][:50])
        fused        = reciprocal_rank_fusion(
            rankings=[bm25_ranking, bi_ranking],
            weights=[0.4, 0.6],
            k=60,
        )
        hybrid_idx  = fused[0][0]
        hybrid_sugg = other_df.iloc[hybrid_idx]["item_description"]
        hybrid_unit = other_df.iloc[hybrid_idx]["unit"]

        rows.append({
            "query_description": query_desc,
            "query_unit":        query_unit,
            "query_source":      query_source,
            "bm25_suggested":    bm25_sugg,
            "bm25_unit":         bm25_unit,
            "bi_suggested":      bi_sugg,
            "bi_unit":           bi_unit,
            "hybrid_suggested":  hybrid_sugg,
            "hybrid_unit":       hybrid_unit,
            "correct_match":     "",   # ← kézzel kitöltendő
        })

    testset = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    testset.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Teszthalmaz mentve: {output_path}")
    print(f"  {len(testset)} sor — töltsd ki a 'correct_match' oszlopot:")
    print("    'bm25'    = a BM25 tippje helyes")
    print("    'bi'      = a Bi-Encoder tippje helyes")
    print("    'hybrid'  = a Hibrid tippje helyes")
    print("    'none'    = egyik sem helyes")
    print("    <szöveg>  = saját helyes leírás (ha egyik sem jó)")

    return testset