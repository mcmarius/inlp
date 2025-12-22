import re
import html
import dateparser
from typing import Optional

def normalize_diacritics(text: str) -> str:
    """
    Standardize Romanian diacritics to comma-below variants.
    ş -> ș
    ţ -> ț
    Ş -> Ș
    Ţ -> Ț
    """
    replacements = {
        "ş": "ș",
        "ţ": "ț",
        "Ş": "Ș",
        "Ţ": "Ț"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def clean_text(text: str) -> str:
    """
    Perform high-level text cleaning:
    1. Decode HTML entities.
    2. Normalize diacritics.
    3. Normalize whitespace.
    """
    if not text:
        return ""
    
    # 1. Decode HTML entities (e.g., &nbsp;)
    # NOTE: &nbsp; doesn't seem to appear in the current dataset, but kept for robustness.
    text = html.unescape(text)
    
    # 2. Normalize diacritics
    text = normalize_diacritics(text)
    
    # 3. Handle special whitespace and multiple spaces
    text = text.replace("\u00a0", " ")  # Non-breaking space
    
    # 4. Collapse multiple spaces and handle the "Word 1 Word 2" vs "Word1Word2" transition artifacts
    # NOTE: The scraper now handles most of this, but we keep it for extra robustness.
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_romanian_date(date_str: str) -> Optional[str]:
    """
    Parse a Romanian date string and return ISO format (YYYY-MM-DD).
    Uses dateparser for robustness.
    """
    if not date_str:
        return None
    
    # dateparser works well with Romanian month names
    dt = dateparser.parse(date_str, languages=['ro'])
    if dt:
        return dt.date().isoformat()
    return None
