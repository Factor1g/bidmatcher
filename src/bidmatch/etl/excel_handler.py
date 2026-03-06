import pandas as pd
from venv import logger
from pathlib import Path
from .excel_sheet_handler import detect_header, detect_columns, normalize_text, parse_excel_sheet
from .description_resolver import resolve_description


MECHANICAL_KEYWORDS = ["gep", "gepeszet", "epuletgepeszet", "mep", "mech", "me", "hvac", "legtechnika", "eg"]

def is_mechanical_sheet(sheet_name: str) -> bool:
    sheet_name_normalized = normalize_text(sheet_name)
    return any(k in sheet_name_normalized for k in MECHANICAL_KEYWORDS)

def parse_excel(file_path: str, project_id: str) -> pd.DataFrame:
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
            df = parse_excel_sheet(file_path = file_path, project_id = project_id, sheet_name = sheet_name)

            if df is None or df.empty:
                continue

             # Csak akkor szűrünk, ha léteznek a szükséges oszlopok
            if 'quantity' in df.columns:
                df = df[df['quantity'].notna()]

            # Legalább az egyik költség típusnak léteznie kell
            cost_cols = [c for c in ['material_cost', 'labor_cost'] if c in df.columns]
            if cost_cols:
                df = df[~df[cost_cols].isna().all(axis=1)]

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
        result_df = resolve_description(concat_df)
    except Exception as e:
        print(f"Hiba az összefűzésnél: {e}")
        result_df = pd.DataFrame()

    return result_df

