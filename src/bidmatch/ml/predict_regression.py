import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bidmatch.config.setting import MODEL_DIR, EMBEDDING_MODEL_NAME, MODEL_DIR_REG

#MODEL_DIR = Path("models/regression_v1")
#EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

material_model = joblib.load(MODEL_DIR_REG / "material_model.pkl")
labor_model = joblib.load(MODEL_DIR_REG / "labor_model.pkl")

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def predict_price(description: str, unit: str):

    text_input = f"{description} | {unit}"

    embedding = embedder.encode(
        [text_input],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    material_price = material_model.predict(embedding)[0]
    labor_price = labor_model.predict(embedding)[0]

    return {
        "material_cost_estimate": float(material_price),
        "labor_cost_estimate": float(labor_price)
    }


if __name__ == "__main__":
    result = predict_price(
        "DN50 horganyzott acélcső szereléssel",
        "m"
    )

    print(result)