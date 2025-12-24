import sys
from pathlib import Path
import importlib.util

# Import stopwords from the word_representations module to avoid duplication
_word_repr_utils_path = Path(__file__).parent.parent / "04_word_representations" / "utils.py"
_spec = importlib.util.spec_from_file_location("word_repr_utils", _word_repr_utils_path)
_word_repr_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_word_repr_utils)

get_improved_stopwords = _word_repr_utils.get_improved_stopwords

__all__ = ['get_improved_stopwords']
