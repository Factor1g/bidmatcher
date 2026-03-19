UNIT_NORMALIZE = {
    "klt": "db",
    "készlet": "db",
    "egys": "db",
    "egység": "db",
    "fm": "m"
}

def normalize_unit(unit: str) -> str:
    u = unit.strip().lower()
    return UNIT_NORMALIZE.get(u, u)