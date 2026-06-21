import numpy as np
import pandas as pd
from pathlib import Path
 
from ..etl.excel_handler import parse_excel
from .hybrid_retriever import HybridRetriever
from ..config.setting import (
    TRAINING_DATA_PATH, EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL,
)

def run_matching(
        input_path: str,
        output_path: str,
        top_k: int = 1,
        top_n_candidates: int = 50,
) -> pd.DataFrame:
    """
    Beolvas egy Excel fájlt, minden sorhoz megkeresi a legjobb egyezést
    a korpuszban, majd visszaírja az eredményt az Excelbe.
 
    Új oszlopok az outputban:
      suggested_item          — legjobb találat leírása
      suggested_unit          — legjobb találat mértékegysége
      suggested_material_cost — legjobb találat anyagköltsége
      suggested_labor_cost    — legjobb találat munkadíja
      match_score             — cross-encoder score
      bi_score                — bi-encoder score
      rrf_score               — RRF fúziós score
    """
    #Korpusz és index betöltése
    print("Korpusz betöltése...")
    corpus_df = pd.read_parquet(TRAINING_DATA_PATH)
    corpus_embeddings = np.load(EMBEDDINGS_PATH)

    #Hibrid retriever inicializálása és index építése
    retriever = HybridRetriever(
        bi_encoder_model_name=EMBEDDING_MODEL_NAME,
        cross_encoder_model_name=CROSS_ENCODER_MODEL,
        bm25_weight=0.4,
        bi_encoder_weight=0.6,
        top_n=top_n_candidates,
        top_k=top_k
    )
    retriever.build_index(corpus_df, corpus_embeddings)

    #Input fájl beolvasása
    print("Input fájl beolvasása...")
    df = parse_excel(file_path=input_path, project_id=0)
 
    if df.empty:
        print("Nem található feldolgozható adat az input fájlban.")
        return pd.DataFrame()
 
    # ── 4. Feldolgozandó sorok szűrése ────────────────────────────────────────
    # parse_excel már kiszűrte a junk sorokat és a quantity=NaN sorokat,
    # de az üres unit-os sorokat explicit kihagyjuk a keresésből
    unit_col = (
        df["unit"].astype(str).str.strip().str.lower()
        if "unit" in df.columns
        else pd.Series("", index=df.index)
    )
 
    processable = df[
        ~unit_col.isin(["", "nan", "none"])
    ].index
 
    total = len(processable)
    print(f"Feldolgozandó sorok: {total}")
 
    # ── 5. Keresés és visszaírás ──────────────────────────────────────────────
    for n, i in enumerate(processable, 1):
        row  = df.loc[i]
        desc = str(row["item_description"]).strip()
        unit = str(row["unit"]).strip()
 
        print(f"  [{n}/{total}] {desc[:60]}")
 
        try:
            results = retriever.search(desc, unit, top_k=top_k)
 
            if results is not None and not results.empty:
                best = results.iloc[0]
                df.at[i, "suggested_item"]          = best.get("item_description", "")
                df.at[i, "suggested_unit"]          = best.get("unit", "")
                df.at[i, "suggested_material_cost"] = best.get("material_cost", np.nan)
                df.at[i, "suggested_labor_cost"]    = best.get("labor_cost", np.nan)
                df.at[i, "match_score"]             = round(float(best.get("cross_encoder_score", 0)), 4)
                df.at[i, "bi_score"]                = round(float(best.get("bi_score", 0)), 4)
                df.at[i, "rrf_score"]               = round(float(best.get("rrf_score", 0)), 6)
            else:
                df.at[i, "suggested_item"] = "UNCERTAIN"
 
        except Exception as e:
            print(f"  Hiba a {i}. sornál: {e}")
 
    # ── 6. Excel mentés ───────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nKész. Output: {output_path}  ({total} sor feldolgozva)")
 
    return df