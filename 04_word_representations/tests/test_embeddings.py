import pytest
import numpy as np
from representations.embeddings import EmbeddingWrapper

@pytest.fixture
def sample_sentences():
    return [
        ["amendă", "aplicat", "comisar", "consumer"],
        ["control", "descoperit", "problemă", "sancțiune"],
        ["amendă", "sancțiune", "mare", "valoare"],
        ["comisar", "verificat", "magazin", "produs"]
    ]

def test_word2vec_training(sample_sentences):
    w2v = EmbeddingWrapper(model_type="word2vec", vector_size=10, min_count=1, window=2)
    w2v.train(sample_sentences)
    
    assert w2v.model is not None
    vec = w2v.get_vector("amendă")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (10,)
    
    similar = w2v.most_similar("amendă", topn=2)
    assert len(similar) <= 2
    assert "word" in similar[0]
    assert "similarity" in similar[0]

def test_fasttext_training(sample_sentences):
    ft = EmbeddingWrapper(model_type="fasttext", vector_size=10, min_count=1, window=2)
    ft.train(sample_sentences)
    
    assert ft.model is not None
    # FastText should handle OOV
    vec = ft.get_vector("super-amendă")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (10,)
    
    similar = ft.most_similar("super-amendă", topn=2)
    assert len(similar) <= 2

def test_invalid_model_type():
    with pytest.raises(ValueError):
        EmbeddingWrapper(model_type="invalid")

def test_untrained_model():
    w2v = EmbeddingWrapper(model_type="word2vec")
    with pytest.raises(ValueError):
        w2v.get_vector("word")
    with pytest.raises(ValueError):
        w2v.most_similar("word")
