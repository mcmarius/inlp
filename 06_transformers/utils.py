import sys
from pathlib import Path
import importlib.util
import pandas as pd
import bertopic

# Monkey-patch BERTopic to fix pandas 3.0+ compatibility
# The offending code in bertopic._bertopic.BERTopic.topics_over_time:
# documents["Timestamps"] = pd.to_datetime(documents["Timestamps"], infer_datetime_format=infer_datetime_format, format=datetime_format)
# 'infer_datetime_format' was removed in pandas 3.0.

original_to_datetime = pd.to_datetime

def patched_to_datetime(*args, **kwargs):
    if "infer_datetime_format" in kwargs:
        kwargs.pop("infer_datetime_format")
    return original_to_datetime(*args, **kwargs)

# We patch pandas.to_datetime globally while bertopic is running topics_over_time 
# OR we can patch the method itself. Patching the method is cleaner.

original_topics_over_time = bertopic._bertopic.BERTopic.topics_over_time

def patched_topics_over_time(self, *args, **kwargs):
    import pandas as pd
    from unittest.mock import patch
    with patch("pandas.to_datetime", patched_to_datetime):
        return original_topics_over_time(self, *args, **kwargs)

bertopic._bertopic.BERTopic.topics_over_time = patched_topics_over_time

# Import stopwords from the word_representations module to avoid duplication
_word_repr_utils_path = Path(__file__).parent.parent / "04_word_representations" / "utils.py"
_spec = importlib.util.spec_from_file_location("word_repr_utils", _word_repr_utils_path)
_word_repr_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_word_repr_utils)

get_improved_stopwords = _word_repr_utils.get_improved_stopwords

__all__ = ['get_improved_stopwords']
