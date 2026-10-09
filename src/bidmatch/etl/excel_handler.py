import pandas as pd
from venv import logger
from pathlib import Path
from .excel_sheet_handler import detect_header, detect_columns, normalize_text, parse_excel_sheet
from .description_resolver import clean_multiline_description, is_junk_row, resolve_description


MECHANICAL_KEYWORDS = ["gep", "gepeszet", "epuletgepeszet", "mep", "mech", "me", "hvac",
                    "legtechnika", "eg", "hutese", "futes", "szelloztetes", "szellozes", "viz",
                    "csatorna", "csatornazas", "suritett levego", "heating", "cooling", "ventilation", "vent", "water", "drainage", "compressed air"]

def is_mechanical_sheet(sheet_name: str) -> bool:
    sheet_name_normalized = normalize_text(sheet_name)
    return any(k in sheet_name_normalized for k in MECHANICAL_KEYWORDS)

def parse_excel(file_path: str, project_id: str, require_cost: bool) -> pd.DataFrame:
    print(f"Parsing excel {file_path.name}")

    file_path = Path(file_path)
    xls = pd.ExcelFile(file_path)

    all_dfs = []

    for sheet_name in xls.sheet_names:
        if not is_mechanical_sheet(sheet_name):
            print(f"Skipping non-mechanical sheet: {sheet_name}")
            continue
        try:
            print(f"Tring to parse: {sheet_name}")
            df = parse_excel_sheet(file_path = file_path, project_id = project_id, sheet_name = sheet_name, drop_incomplete = False)

            if df is None or df.empty:
                continue

            if "item_description" in df.columns:
                df["item_description"] = df["item_description"].apply(
                    clean_multiline_description
                )

            if "item_description" in df.columns:
                junk_mask = df.apply(   
                    lambda r : is_junk_row(r["item_description"], r.get("quantity") if "quantity" in df.columns else None), axis=1
                )
                n_junk = junk_mask.sum()
                if n_junk > 0:
                    print(f"  Skipping {n_junk} junk rows in sheet {sheet_name}")   
                df = df[~junk_mask]

            if df.empty:
                continue

            # ── 4. resolve_description MUNKALAPОНKÉNT ────────────────────────
            # Kritikus: teljes, szűretlen sorokon fut, hogy a kontextus
            # (last_full_desc) ne szakadjon meg ár/unit nélküli soroknál.
            df = resolve_description(df)

            print(f"Rows in sheet {sheet_name}: {len(df)}")

            all_dfs.append(df)

        except Exception as e:
            print(f"Error processing sheet {sheet_name} in {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_dfs:
        print("Nem található feldolgozható munkalap.")
        return pd.DataFrame()

    try:
        concat_df = pd.concat(all_dfs, ignore_index=True)
        
    except Exception as e:
        print(f"Hiba az összefűzésnél: {e}")
        result_df = pd.DataFrame()

    result_df = _filter_for_corpus(concat_df, require_cost=require_cost)

    print(f"Total rows after filtering: {len(result_df)}")

    return result_df


def _filter_for_corpus(df: pd.DataFrame, require_cost: bool) -> pd.DataFrame:
    """
        Korpusz építéshez szükséges szűrés — CSAK a resolve_description után hívandó.
 
        Kiszűri azokat a sorokat amelyeknek:
      - nincs item_description
      - nincs quantity
      - nincs unit (vagy üres)
      - nincs se material_cost se labor_cost
    """
    if "quantity" in df.columns:
        df = df[df["quantity"].notna()]

    if "unit" in df.columns:
        unit_clean = df["unit"].astype(str).str.strip().str.lower()
        df = df[~unit_clean.isin(["", "nan", "none"])]

    if require_cost:
        cost_cols = [c for c in ["material_cost", "labor_cost"] if c in df.columns]
        if cost_cols:
            df = df[~df[cost_cols].isna().all(axis=1)]

    return df.reset_index(drop=True)