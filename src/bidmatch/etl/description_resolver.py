import re
import pandas as pd

UI_PATTERN = re.compile(r"^\s*ua\.?,?\s*de\s*", re.IGNORECASE)
SIZE_ONLY_PATTERN = re.compile(r"^\s*(D|DN)\s*\d+\s*$", re.IGNORECASE)

def resolve_description(df: pd.DataFrame, desc_col = "item_description"):
    last_full_desc = None
    resolved = []

    for desc in df[desc_col]:
        if not isinstance(desc, str) or not desc.strip():
            resolved.append(desc)
            continue
    
        normalized = desc.strip()

        if UI_PATTERN.match(normalized):
            size_part = UI_PATTERN.sub("", normalized).strip()
            if last_full_desc:
                resolved.append(f"{last_full_desc} {size_part}")
            else:
                new_desc = normalized
            resolved.append(size_part)

        elif SIZE_ONLY_PATTERN.match(normalized):
            if last_full_desc:
                new_desc = f"{last_full_desc} {normalized}"
            else:
                new_desc = normalized
            resolved.append(new_desc)

        else:
            last_full_desc = normalized
            resolved.append(normalized)

    df[desc_col] = resolved
    return df   