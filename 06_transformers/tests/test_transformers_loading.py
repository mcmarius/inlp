import pytest
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Ensure we can import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import get_improved_stopwords

def test_stopwords_load():
    sw = get_improved_stopwords()
    assert "anpc" in sw
    assert "de" in sw
    assert len(sw) > 20

def test_model_names_valid():
    # We don't necessarily want to download 1GB models in CI, so we just check if the strings are correct
    # or try to load tokenizer which is small.
    t5_name = "dumitrescustefan/t5-v1_1-base-romanian"
    bert_name = "racai/distilbert-base-romanian-cased"
    
    # Just try to load tokenizers to verify model ID validity
    try:
        AutoTokenizer.from_pretrained(t5_name)
        AutoTokenizer.from_pretrained(bert_name)
    except Exception as e:
        pytest.fail(f"Could not load tokenizer for valid models: {e}")
