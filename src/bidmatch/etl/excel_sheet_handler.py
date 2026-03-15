from venv import logger
import pandas as pd
import unicodedata
from pathlib import Path
from .unit_normalizer import normalize_unit

COLUMN_KEYWORDS = {
    "item_description" : ["megnevezes" , "tetel szovege" , "tetel" , "tetel megnevezese" , "item" , "description" , "leiras" , "tetel leirasa"],
    "quantity": ["mennyiseg" , "menny" , "menny." , "qty" , "quantity" , "darab"],
    "material_cost": ["anyag egysegar", "egysegar anyag", "anyag egysegre", "anyag"],
    "labor_cost": ["dij egysegar", "egysegar dij", "munkadij", "szereles", "dij egysegre", "dij"],
    "unit": ["mertekegyseg" , "mert.egys" , "egyseg" , "mertek" , "unit", "me.", "egys", "egys."]
}

FORBIDDEN_KEYWORDS = {
    "unit": ["anyag", "dij", "osszesen", "ar"],
    "material_cost": ["osszesen", "total"],
    "labor_cost": ["osszesen", "total"],
    "quantity": ["ar", "egysegar", "osszesen"]
}

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.strip()

def detect_header(df: pd.DataFrame) -> int:
    header_row = 0
    max_score = 0
    for i in range(min(10,len(df))):
        row = df.iloc[i].astype(str)
        score = 0
        for cell in row:
            cell_normalized = normalize_text(cell)
            for keyword_list in COLUMN_KEYWORDS.values():
                if any(k in cell_normalized for k in keyword_list):
                    score += 1
        if score > max_score:
            max_score = score
            header_row = i
    return header_row

def detect_columns(columns) -> dict:
    detected = {}
    used_cols = set()
    assigned_keys = set()

    potential_matches = []
    for key, keyword_list in COLUMN_KEYWORDS.items():
        forbidden_words = FORBIDDEN_KEYWORDS.get(key, [])
        for col in columns:
            col_normalized = normalize_text(col)
            if any(f in col_normalized for f in forbidden_words):
                continue
        
            for keyword in keyword_list:
                if keyword in col_normalized:
                    coverage = len(keyword) / len(col_normalized) if len(col_normalized) > 0 else 0
                    score = (coverage *100) + len(keyword)

                    potential_matches.append({
                        "key": key,
                        "column": col,
                        "score": score
                    })
        
    potential_matches.sort(key=lambda x: x["score"], reverse=True)

    for match in potential_matches:
        if match["column"] not in used_cols and match["key"] not in assigned_keys:
            detected[match["key"]] = match["column"]
            used_cols.add(match["column"])
            assigned_keys.add(match["key"])

    return detected       

def parse_excel_sheet(file_path: str, project_id: str , sheet_name: str) -> pd.DataFrame:    

    file_path = Path(file_path)
    print(f"Reading sheet: {file_path.name} / {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    print(f"Excel sheet read: {file_path.name} / {sheet_name} - {len(df)} sor")
    detect_header_row = detect_header(df)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=detect_header_row)
    column_map = detect_columns(df.columns)
    print(f"Columns detected: {column_map}")

    for key in COLUMN_KEYWORDS.keys():
        if key not in column_map:
            print(f"Column for '{key}' not detected in {file_path.name} / {sheet_name}")
            column_map[key] = None
    
    df = df[list(column_map.values())]
    df = df.rename(columns={v: k for k, v in column_map.items()})

    df = df.dropna(subset=["item_description", "quantity"], how="any")

    if "unit" in df.columns:
        df["unit"] = df["unit"].astype(str).apply(normalize_unit)
    if "quantity" in df:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "material_cost" in df:
        df["material_cost"] = pd.to_numeric(df["material_cost"], errors="coerce")
    if "labor_cost" in df:
        df["labor_cost"] = pd.to_numeric(df["labor_cost"], errors="coerce")
    
    df["project_id"] = project_id
    df["source_file"] = file_path.name
    df["source_sheet"] = sheet_name

    return df.reset_index(drop=True)