"""
excel_price_suggester.py
------------------------
Beolvas egy ár nélküli Excel-t, minden sorára megkeresi a legjobb egyező
historikus tételt (bi-encoder + cross-encoder), majd visszaír egy új Excel-t
az ajánlott anyag- és díjárakkal.

Használat:
    python excel_price_suggester.py input.xlsx output.xlsx
    python excel_price_suggester.py input.xlsx  # output: input_priced.xlsx
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from ..etl.description_resolver import resolve_description

from ..etl.excel_sheet_handler import detect_header, detect_columns
from ..ml.test_similarity import build_artifacts, build_query_text, search

def read_input_excel(file_path: Path) -> tuple[pd.DataFrame, dict]:
    print(f"Reading input Excel: {file_path}")
    df = pd.read_excel(file_path, header=None)
    header_row = detect_header(df)
    df = pd.read_excel(file_path, header=header_row)
    column_map = detect_columns(df.columns)
    print(f"Detected columns: {column_map}")

    df = df.rename(columns={v: k for k, v in column_map.items()})

    if "item_description" not in df.columns:
        raise ValueError("No 'item_description' column in the file!")
    
    df = resolve_description(df)

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    return df, column_map

def process_excel(input_path: Path, output_path: Path):
    df, column_map = read_input_excel(input_path)
    
    corpus_df, corpus_embeddings, bi_encoder, cross_encoder = build_artifacts()

    df["suggested_item"] = ""
    df["suggested_unit"] = ""
    df["material_cost"]  = np.nan
    df["labor_cost"]     = np.nan
    df["match_score"]    = np.nan
    df["bi_score"]       = np.nan

    total = len(df)
    for i, row in df.iterrows():
        desc = str(row.get("item_description", "")).strip()
        unit = str(row.get("unit", "")).strip() if "unit" in df.columns else ""

        if not desc or desc.lower() in ("nan", "none", ""):
            continue
        
        if unit.lower() in ("nan", "none", ""):
            continue

        print(f"Processing row {i+1}/{total}: {desc[:60]}")

        try:
            result = search(desc, unit, corpus_df, corpus_embeddings, bi_encoder, cross_encoder, top_k=1, top_n_candidates=50)
            if not result.empty:
                best = result.iloc[0]
                df.at[i, "suggested_item"] = best.get("item_description", "")
                df.at[i, "suggested_unit"] = best.get("unit", "")
                df.at[i, "material_cost"]  = best.get("material_cost", np.nan)
                df.at[i, "labor_cost"]     = best.get("labor_cost", np.nan)
                df.at[i, "match_score"]    = round(float(best.get("cross_encoder_score", 0)), 4)
                df.at[i, "bi_score"]       = round(float(best.get("bi_encoder_score", 0)), 4)
        except Exception as e:
            print(f"Error processing row {i+1}: {e}")

    print(df.head(30))
    df.to_excel(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Excel Price Suggester")
    parser.add_argument("input_excel", type=Path, help="Path to input Excel file")
    parser.add_argument("output_excel", type=Path, nargs="?", help="Path to output Excel file (optional)")
    args = parser.parse_args()

    input_path = args.input_excel
    output_path = args.output_excel if args.output_excel else input_path.with_name(input_path.stem + "_priced.xlsx")

    process_excel(input_path, output_path)