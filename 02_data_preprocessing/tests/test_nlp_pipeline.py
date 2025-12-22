import pytest
from preprocessing.nlp_pipeline import RomanianNLP, get_romanian_stopwords

@pytest.fixture(scope="module")
def nlp():
    return RomanianNLP()

def test_stanza_processing(nlp):
    text = "Consumatorii au drepturi."
    result = nlp.process_with_stanza(text)
    assert len(result) > 0
    # Check for lemmatization
    lemmas = [token["lemma"] for token in result]
    assert "consumator" in lemmas

def test_spacy_processing(nlp):
    text = "Consumatorii au drepturi."
    result = nlp.process_with_spacy(text)
    assert len(result) > 0
    # SpaCy lemmatization
    lemmas = [token["lemma"] for token in result]
    assert "consumator" in lemmas

def test_nltk_stemming(nlp):
    text = "Consumatorii au drepturi."
    result = nlp.process_with_nltk(text)
    assert len(result) > 0
    # NLTK stemming for Romanian is aggressive
    stems = [token["stem"] for token in result]
    # 'consumatori' -> 'consum' (aggressive stemming)
    assert "consum" in stems

def test_stemming_vs_lemmatization(nlp):
    text = "Am cumpărat niște mere."
    diff = nlp.compare_stemming_vs_lemmatization(text)
    assert len(diff) > 0
    for item in diff:
        assert "text" in item
        assert "stem" in item
        assert "lemma" in item

def test_stopwords():
    stopwords = get_romanian_stopwords()
    # NLTK Romanian stopwords are often diacritic-less
    assert "si" in stopwords
    assert "dupa" in stopwords
