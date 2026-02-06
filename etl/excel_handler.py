import pandas as pd
import unicodedata
from pathlib import Path

COLUMN_KEYWORDS = {
    "item_description" : ["megnevezes" , "tetel szovege" , "tetel" , "tetel megnevezese" , "item" , "description" , "leiras" , "tetel leirasa"],
    "quantity": ["mennyiseg" , "menny" , "qty" , "quantity" , "darab"],
    "material_cost": ["anyag egysegar", "egysegar anyag", "anyag egysegre", "anyag"],
    "labor_cost": ["dij egysegar", "egysegar dij", "munkadij", "szereles", "dij egysegre", "dij"],
    "unit": ["mertekegyseg" , "mert.egys" , "egyseg" , "mertek" , "unit", "me.", "egys", "egys."]
}

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

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
