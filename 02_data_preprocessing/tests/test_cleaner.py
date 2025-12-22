import pytest
from preprocessing.cleaner import clean_text, normalize_diacritics

def test_normalize_diacritics():
    assert normalize_diacritics("peste tot în ţară cu şase") == "peste tot în țară cu șase"
    assert normalize_diacritics("Ţara Ştefan") == "Țara Ștefan"

def test_clean_text():
    raw = "  Aceasta este   o stire&nbsp;cu diacritice şchioape.  "
    expected = "Aceasta este o stire cu diacritice șchioape."
    assert clean_text(raw) == expected
