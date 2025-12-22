import pytest
from analysis.topic_modeling import TopicModeler, BM25Searcher

def test_lda_modeling():
    docs = [
        "pește proaspăt la magazinul din colț",
        "alimente expirate găsite de comisari",
        "control anpc la raionul de pește",
        "amenzi pentru produse neconforme și expirate",
        "comisarii anpc au închis magazinul alimentar"
    ]
    modeler = TopicModeler(n_topics=2)
    modeler.fit_lda(docs)
    topics = modeler.get_topics('lda', n_top_words=3)
    assert len(topics) == 2
    assert all(len(words) == 3 for words in topics.values())

def test_bm25_search():
    docs = [
        ["anpc", "control", "magazine"],
        ["pește", "proaspăt", "alimente"],
        ["amenzi", "expirate", "control"]
    ]
    searcher = BM25Searcher(docs)
    results = searcher.search(["control"], n=2)
    assert len(results) == 2
    # Document 0 and 2 contain "control"
    assert set(results) == {0, 2}
