from pathlib import Path
# PROJECT ROOT 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# DATA 
DATA_SOURCE = PROJECT_ROOT / "data/raw"
TRAINING_DATA_PATH = PROJECT_ROOT / "data/training/bid_items.parquet"

# MODEL 
MODEL_DIR = PROJECT_ROOT / "models/bid_pricing"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.npy"
MODEL_DIR_REG = MODEL_DIR / "regression"

#  ML 
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5