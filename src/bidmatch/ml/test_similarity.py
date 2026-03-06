from ..config.setting import DATA_SOURCE, TRAINING_DATA_PATH, EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME, TOP_K
from ..ml.build_traning_dataset import build_big_dataframe, build_text_input, generate_embeddings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def build_artifacts():
    df = build_big_dataframe(DATA_SOURCE)
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return df, embeddings, model

def search(query_text: str):
    df, embeddings, model = build_artifacts()
    
    query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = np.argsort(similarities)[-TOP_K:][::-1]

    results = df.iloc[top_idx].copy()
    results["similarity"] = similarities[top_idx]

    return results[[
        "item_description",
        "unit",
        "material_cost",
        "labor_cost",
        "similarity"
    ]]


if __name__ == "__main__":
    query = "DN50 horganyzott acélcső szereléssel"
    results = search(query)

    print("\nTop similar items:\n")
    print(results)
