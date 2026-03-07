from ..config.setting import DATA_SOURCE, TRAINING_DATA_PATH, EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME, TOP_K, CROSS_ENCODER_MODEL
from ..ml.build_traning_dataset import build_big_dataframe, build_text_input, generate_embeddings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

TOP_N_CANDIDATES = 50

def build_artifacts():
    #df = build_big_dataframe(DATA_SOURCE)
    df = pd.read_parquet(TRAINING_DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    #model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return df, embeddings

def build_query_text(description: str, unit: str) -> str:
    """
    Mirror the same text format used during embedding generation:
      'item_description | unit'
    If unit is unknown at query time, leave it empty.
    """
    unit = unit.strip() if unit else ""
    return f"{description.strip()} | {unit}" if unit else description.strip()

def search(
        query_text: str, unit: str = "",
        top_k: int = TOP_K, top_n_candidates: int = TOP_N_CANDIDATES,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        cross_encoder_model: str = CROSS_ENCODER_MODEL
        ) -> pd.DataFrame:
    
    """
    Two-stage retrieval:
      Stage 1 – Bi-Encoder cosine similarity → top_n_candidates
      Stage 2 – Cross-Encoder re-ranking     → top_k final results

    Args:
        query_text: raw item description to search for
        unit: optional measurement unit (e.g. 'm', 'db', 'fm')
        top_k: number of final results to return
        top_n_candidates: candidate pool size for stage 1
        bi_encoder_model: HuggingFace model id for bi-encoder
        cross_encoder_model: HuggingFace model id for cross-encoder

    Returns:
        DataFrame with top_k results sorted by cross-encoder score
    """

    df, corpus_embeddings = build_artifacts()
    
    print(f"[Bi-Encoder] Loading {embedding_model} …")
    bi_encoder = SentenceTransformer(embedding_model)

    formatted_query = build_query_text(query_text, unit)

    query_embedding = bi_encoder.encode([formatted_query], convert_to_numpy=True, normalize_embeddings=True)

    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    candidate_idx = np.argsort(similarities)[-top_n_candidates:][::-1]

    candidate_df = df.iloc[candidate_idx].copy()
    candidate_df["bi_encoder_score"] = similarities[candidate_idx]

    print(f"[Bi-Encoder] Top-{top_n_candidates} candidates retrieved.")

    print(f"[Cross-Encoder] Loading {cross_encoder_model} …")
    cross_encoder = CrossEncoder(cross_encoder_model)

    candidate_texts = (candidate_df["item_description"].astype(str).str.strip() + " | " + candidate_df["unit"].astype(str).str.strip()).tolist()

    pairs = [(query_text, cand) for cand in candidate_texts]
    ce_scores = cross_encoder.predict(pairs)

    candidate_df["cross_encoder_score"] = ce_scores

    result_df = (
        candidate_df
        .sort_values("cross_encoder_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    print(f"[Cross-Encoder] Re-ranking done. Returning top-{top_k} results.")

    return result_df[[
        "item_description",
        "unit",
        "material_cost",
        "labor_cost",
        "bi_encoder_score",
        "cross_encoder_score",
    ]]


if __name__ == "__main__":
    query = "DN50 horganyzott acélcső szereléssel"
    results = search(query)

    print("\nTop similar items:\n")
    print(results)
