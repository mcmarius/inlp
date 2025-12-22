import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

class TopicModeler:
    def __init__(self, n_topics: int = 5, stopwords: List[str] = None):
        self.n_topics = n_topics
        self.stopwords = self._fix_stopwords(stopwords)
        self.lda_model = None
        self.nmf_model = None
        self.vectorizer_lda = None
        self.vectorizer_nmf = None

    def _fix_stopwords(self, stopwords: List[str]) -> List[str]:
        """Ensures stopwords are consistent with scikit-learn's default tokenizer."""
        if not stopwords:
            return None
        
        import re
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        fixed = set()
        for word in stopwords:
            tokens = token_pattern.findall(word.lower())
            if tokens:
                fixed.update(tokens)
            else:
                fixed.add(word.lower())
        return list(fixed)

    def fit_lda(self, documents: List[str]):
        """Fit LDA model on documents."""
        self.vectorizer_lda = CountVectorizer(
            max_df=0.95, 
            min_df=2, 
            stop_words=self.stopwords
        )
        dtm = self.vectorizer_lda.fit_transform(documents)
        self.lda_model = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        self.lda_model.fit(dtm)

    def fit_nmf(self, documents: List[str]):
        """Fit NMF model on documents."""
        self.vectorizer_nmf = TfidfVectorizer(
            max_df=0.95, 
            min_df=2, 
            stop_words=self.stopwords
        )
        tfidf = self.vectorizer_nmf.fit_transform(documents)
        self.nmf_model = NMF(n_components=self.n_topics, random_state=42, init='nndsvd')
        self.nmf_model.fit(tfidf)

    def get_topics(self, model_type: str = 'lda', n_top_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each topic."""
        if model_type == 'lda':
            model = self.lda_model
            vectorizer = self.vectorizer_lda
        else:
            model = self.nmf_model
            vectorizer = self.vectorizer_nmf

        if model is None:
            return {}

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics[topic_idx] = top_features
        return topics

class BM25Searcher:
    def __init__(self, documents: List[List[str]]):
        """
        documents: List of tokenized documents.
        """
        self.bm25 = BM25Okapi(documents)
        self.documents = documents

    def search(self, query: List[str], n: int = 5) -> List[int]:
        """Returns indices of top n documents matching the query."""
        scores = self.bm25.get_scores(query)
        top_n_indices = scores.argsort()[:-n - 1:-1]
        return top_n_indices.tolist()
