import pandas as pd

UNIT_NORMALIZE = {
    "klt": "db",
    "készlet": "db",
    "egys": "db",
    "egység": "db",
    "fm": "m"
}

def normalize_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return ""
    u = unit.strip().lower()
    if not u or u in ["nan", "none"]:
        return ""
    return UNIT_NORMALIZE.get(u, u)