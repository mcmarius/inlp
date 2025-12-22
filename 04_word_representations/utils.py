import sys
from pathlib import Path

# Add project root to path to access other modules if needed
sys.path.append(str(Path(__file__).parent.parent))

# Can't directly import from '02_data_preprocessing' as it starts with a digit
# We will assume stopwords are either passed or we use a sensible default
def get_romanian_stopwords():
    # Minimal set of Romanian stopwords if the other module is not accessible
    return ["de", "la", "și", "un", "o", "în", "pe", "din", "cu", "că", "să", "a", "al", "ai", "ale"]

def get_improved_stopwords():
    """Returns a refined list of Romanian stopwords including domain-specific terms."""
    base_stopwords = get_romanian_stopwords()
    custom_domain_stopwords = [
        "anpc", "comisar", "control", "magazine", "produs", "lei", "amendă", 
        "comercializă", "urmare", "respectare", "principal", "constatat", 
        "neconformitate", "neconformităților", "oprire", "lipsă", "sine",
        "fost", "avut", "care", "un", "una", "această", "este", "sunt",
        "prin", "pentru", "tot", "toate", "acest", "aceasta", "după", "fără",
        "și", "al", "la", "pe", "sau", "să", "din", "de", "cu", "în", "la"
    ]
    # Remove duplicates and ensure all are lowercase
    combined = set(word.lower() for word in base_stopwords + custom_domain_stopwords)
    return list(combined)
