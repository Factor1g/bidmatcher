import re
import unicodedata

HU_STOPWORDS = {
    "a", "az", "és", "vagy", "de", "hogy", "nem", "is", "meg", "csak",
    "már", "még", "el", "fel", "be", "ki", "le", "össze", "szét",
    "ha", "mert", "mint", "ami", "aki", "amely", "amelynek", "szerint",
    "valamint", "illetve", "ill", "kompletten", "felszerelve", "beépítve",
    "elhelyezve", "szerelve", "szállítva", "együtt", "nélkül", "tartozó"
}

# DN/NÁ normalizáló: "NÁ 50" → "DN50", "DN 50" → "DN50"
DN_NORM = re.compile(r"\b(DN|NÁ|NA|D)\s*(\d+)\b", re.IGNORECASE)

def normalize_dn(text: str) -> str:
    return DN_NORM.sub(lambda m: f"DN{m.group(2)}", text)

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    if not isinstance(text, str):
        return []
    
    text = normalize_dn(text)
    text = normalize_unicode(text).lower()

    tokens = re.findall(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in HU_STOPWORDS]

    return tokens

