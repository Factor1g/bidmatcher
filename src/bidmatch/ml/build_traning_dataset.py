from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ..etl.excel_handler import parse_excel
from ..config.setting import DATA_SOURCE, TRAINING_DATA_PATH, EMBEDDINGS_PATH, EMBEDDING_MODEL_NAME


# DATA_SOURCE = Path("data/raw/")
# TRAINING_DATA_PATH = Path("data/training/bid_items.parquet")
# EMBEDDINGS_PATH = Path("models/bid_pricing/embeddings.npy")

# EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def get_all_excels(src_dir: Path) -> list[Path]:
    return list(src_dir.rglob("*.xlsx")) + list(src_dir.rglob("*.xls"))

def build_big_dataframe(src_dir: Path) ->pd.DataFrame:
    excel_files = get_all_excels(src_dir)

    if not excel_files:
        raise ValueError(f"Excel files can not be found in given directory: {src_dir}!")
    
    all_dfs = []

    for idx, file_path in enumerate(excel_files):
        print(f"Processing {file_path.name} ({idx+1} / {len(excel_files)})")

        try:
            df = parse_excel(file_path, idx+1)

            if df is not None and not df.empty:
                all_dfs.append(df)

        except Exception as e:
            print(f"Error during parsing {file_path.name}: {e}")

    if not all_dfs:
        raise ValueError("No parseble excel found!")

    result_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nAll rows from xls: {len(result_df)}")    

    return result_df

def build_text_input(df: pd.DataFrame) -> pd.Series:
    return ( df["item_description"].astype(str).str.strip()
    + " | "
    + df["unit"].astype(str).str.strip() )

def generate_embeddings(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

def build_training_pipeline():
    print("Building dataset...")
    df = build_big_dataframe(DATA_SOURCE)

    print(f"Rows before cleaning: {len(df)}")

    #Kötelező oszlop ellenőrzés
    required_cols = ["item_description", "unit"]
    df = df.dropna(subset=required_cols)

    #String normalizálás
    df["item_description"] = df["item_description"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()

    #Üres string szűrés
    df = df[
        (df["item_description"] != "") &
        (df["unit"] != "")
    ]

    print(f"Rows after cleaning: {len(df)}")

    print("Building text input...")
    df["text_input"] = build_text_input(df)

    print("Generating embeddings...")
    embeddings = generate_embeddings(df["text_input"].tolist()) 

    print("Saving dataset to Parquet...")
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRAINING_DATA_PATH, index=False)    

    print("Saving embeddings...")
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    print("Full training dataset + embeddings built successfully.")


if __name__ == "__main__":
    build_training_pipeline()