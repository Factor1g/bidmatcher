import re
import pandas as pd

JUNK_PATTERN = re.compile(
    r"^(megjegyzés|a költségvetés|tétel elnevezés|munkalap összesítő|"
    r"must have|not clearly|other,|általános kiírás|összesen:|összesítő:)",
    re.IGNORECASE)

# "ua. de DN50" / "u.a., de DN50"
UI_PATTERN = re.compile(r"^\s*ua\.?,?\s*de\s*", re.IGNORECASE)

# DN / NÁ / D + szám, opcionális zárójelben csőmérettel és/vagy utótag:
# DN50, NÁ 32, DN 50 (54x1,5 mm), DN 125 méretben, DN80
SIZE_DN_PATTERN = re.compile(
    r"^\s*(D|DN|NÁ|NA)\s*\d+([.,]\d+)?(\s*\([\d.,xX\s\w]+\))?(\s*méretben)?\s*$",
    re.IGNORECASE
)

# DN + kötőjel + csőméret: "DN 25 - 28x1,2 mm", "DN 50 - 54,0x1,5 mm"
SIZE_DN_PIPE_PATTERN = re.compile(
    r"^\s*(DN|NÁ|NA|D)\s*\d+\s*[-–]\s*\d+([.,]\d+)?\s*[xX]\s*\d+([.,]\d+)?(\s*mm)?\s*$",
    re.IGNORECASE
)

# Átmérő: Ø20mm, Ø 22,02 mm, ø40-es csőre (tizedes is megengedett)
SIZE_DIAMETER_PATTERN = re.compile(
    r"^\s*[Øø]\s*\d+([.,]\d+)?(\s*mm)?(-es|-as)?(\s*csőre)?\s*$",
    re.IGNORECASE
)

# Cső méret: 42x1,5mm, 22 x 1,2 mm, 32x3,0mm csővezetékre, 110x15.10
# szóköz megengedett az x körül, tizedes az első számnál is
SIZE_PIPE_PATTERN = re.compile(
    r"^\s*\d+([.,]\d+)?\s*[xX]\s*\d+([.,]\d+)?(\s*mm)?(\s*csővezetékre)?\s*$",
    re.IGNORECASE
)

# átm. 32 mm, átm. 40 mm
SIZE_ATM_PATTERN = re.compile(
    r"^\s*átm\.?\s*\d+([.,]\d+)?\s*(mm)?\s*$",
    re.IGNORECASE
)

# Körös: 3 körös, 6 körös
SIZE_KOROS_PATTERN = re.compile(r"^\s*\d+\s*körös\s*$", re.IGNORECASE)

# Termékkód: SR 100 3000, SR125 3000
SIZE_PRODUCT_CODE_PATTERN = re.compile(
    r"^\s*[A-Za-z]{2,5}[\dA-Za-z]*[-\s]?\d{1,4}(\s+\d+){0,2}\s*$"
)

# Komplex cikkszám: AF-15-D/625x125, AF-15-D/3000x125 (szalagkivitel)
SIZE_ARTICLE_NUMBER_PATTERN = re.compile(
    r"^\s*[A-Za-z0-9]{1,6}[-/][A-Za-z0-9][-A-Za-z0-9/xX*]+\s*(\(.*\))?\s*$"
)

SIZE_PATTERNS = [
    SIZE_DN_PATTERN,
    SIZE_DN_PIPE_PATTERN,
    SIZE_ATM_PATTERN,
    SIZE_DIAMETER_PATTERN,
    SIZE_PIPE_PATTERN,
    SIZE_KOROS_PATTERN,
    SIZE_PRODUCT_CODE_PATTERN,
    SIZE_ARTICLE_NUMBER_PATTERN,
]

def is_junk_row(text: str, quantity: None) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    if JUNK_PATTERN.match(text.strip()):
        return True
    if quantity is None or (isinstance(quantity, float) and pd.isna(quantity)):
        if len(text.strip()) < 50:  # Rövid, értelmetlen leírások gyakran szemét sorok
            return True
    return False

def clean_multiline_description(text: str) -> str:
    """
    Newline-os leírások kezelése — a \n utáni rész méretsorként viselkedik.

    Példa:
      "Viega ötrétegű műanyag csővezeték\n25x2,8 mm"
      → "Viega ötrétegű műanyag csővezeték 25x2,8 mm"

    A description_resolver ELŐTT kell futtatni.
    """
    if not isinstance(text, str):
        return text
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    if len(parts) <= 1:
        return text
    return ' '.join(parts)


def is_size_only(text: str) -> bool:
    for pattern in SIZE_PATTERNS:
        if pattern.match(text):
            return True
    return False

def resolve_description(df: pd.DataFrame, desc_col = "item_description"):
    last_full_desc = None
    resolved = []

    qty_col = "quantity" if "quantity" in df.columns else None

    for i, row in df.iterrows():
        desc = row[desc_col]
        qty = row[qty_col] if qty_col else None
        if not isinstance(desc, str) or not desc.strip():
            resolved.append(desc)
            continue
    
        normalized = clean_multiline_description(desc).strip()

        if UI_PATTERN.match(normalized):
            # "ua. de DN50" -> előző leírás + az utána jövő méretrész
            size_part = UI_PATTERN.sub("", normalized).strip()
            new_desc = f"{last_full_desc} {size_part}" if last_full_desc else size_part
            resolved.append(new_desc)

        elif is_size_only(normalized):
            # Puszta méret/kód sor -> örökli az előző teljes leírást
            new_desc = f"{last_full_desc} {normalized}" if last_full_desc else normalized
            resolved.append(new_desc)

        else:
            # Teljes leírás -> eltároljuk referenciának
            last_full_desc = normalized
            resolved.append(normalized)

    df[desc_col] = resolved
    return df   