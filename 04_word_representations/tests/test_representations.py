import pytest
import numpy as np
from representations.bow import BoWRepresentation
from representations.tfidf import TFIDFRepresentation
from representations.bm25 import BM25Representation

@pytest.fixture
def sample_docs():
    return [
        "ana are mere",
        "ion are pere",
        "ana si ion au fructe",
        "merele si perele sunt fructe"
    ]

def test_bow(sample_docs):
    bow = BoWRepresentation()
    dtm = bow.fit_transform(sample_docs)
    assert dtm.shape[0] == 4
    assert len(bow.feature_names) > 0
    top_words = bow.get_top_n_words(1)
    # 'are' appears twice, 'ana' twice, 'ion' twice. 
    # Depending on tokenization and min_df/max_features.
    # CountVectorizer default: r"(?u)\b\w\w+\b" (words of at least 2 chars)
    # 'are', 'ana', 'ion' are all 3 chars.
    assert any(w[1] >= 2 for w in top_words)

def test_tfidf(sample_docs):
    tfidf = TFIDFRepresentation()
    matrix = tfidf.fit_transform(sample_docs)
    assert matrix.shape[0] == 4
    top_words = tfidf.get_top_tfidf_words(0, n=2)
    assert len(top_words) == 2
    assert isinstance(top_words[0][1], float)

def test_bm25():
    tokenized_corpus = [
        ["ana", "are", "mere"],
        ["ion", "are", "pere"],
        ["ana", "si", "ion", "au", "fructe"]
    ]
    bm25 = BM25Representation(tokenized_corpus)
    query = ["ana"]
    scores = bm25.get_scores(query)
    assert scores[0] > scores[1] # 'ana' matches doc 0 but not doc 1
    assert scores[0] > 0
    top_indices = bm25.get_top_n(query, n=1)
    assert top_indices[0] in [0, 2]
