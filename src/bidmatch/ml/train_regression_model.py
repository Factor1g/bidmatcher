from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from bidmatch.config.setting import  EMBEDDING_MODEL_NAME, MODEL_DIR_REG, TRAINING_DATA_PATH

# ---------------- CONFIG ----------------

# DATA_PATH = Path("data/training/bid_items_v2.parquet")

MODEL_DIR_REG.mkdir(parents=True, exist_ok=True)
# EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"



# ---------------- LOAD DATA ----------------

df = pd.read_parquet(TRAINING_DATA_PATH)

df = df.dropna(subset=["material_cost", "labor_cost"])

df["text_input"] = (
    df["item_description"].astype(str) + " | " +
    df["unit"].astype(str)
)

# ---------------- EMBEDDING ----------------

print("Generating embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

X = embedder.encode(
    df["text_input"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

y_material = df["material_cost"].values
y_labor = df["labor_cost"].values


# ---------------- SPLIT ----------------

X_train, X_test, y_mat_train, y_mat_test = train_test_split(
    X, y_material, test_size=0.2, random_state=42
)

_, _, y_lab_train, y_lab_test = train_test_split(
    X, y_labor, test_size=0.2, random_state=42
)


# ---------------- TRAIN ----------------

print("Training material cost model...")
material_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

material_model.fit(X_train, y_mat_train)


print("Training labor cost model...")
labor_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

labor_model.fit(X_train, y_lab_train)


# ---------------- EVALUATE ----------------

mat_pred = material_model.predict(X_test)
lab_pred = labor_model.predict(X_test)

print("Material MAE:", mean_absolute_error(y_mat_test, mat_pred))
print("Labor MAE:", mean_absolute_error(y_lab_test, lab_pred))


# ---------------- SAVE ----------------

joblib.dump(material_model, MODEL_DIR_REG / "material_model.pkl")
joblib.dump(labor_model, MODEL_DIR_REG / "labor_model.pkl")

print("Regression models saved.")