import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from utils import get_improved_stopwords

class TFIDFRepresentation:
    def __init__(self, max_features: int = 1000, stop_words: List[str] = None):
        if stop_words is None:
            stop_words = get_improved_stopwords()
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.feature_names = None
        self.tfidf_matrix = None

    def fit_transform(self, documents: List[str]):
        """Creates the TF-IDF matrix."""
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix

    def get_top_tfidf_words(self, doc_idx: int, n: int = 10) -> List[Tuple[str, float]]:
        """Returns the top n words by TF-IDF score for a specific document."""
        row = self.tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = row.argsort()[-n:][::-1]
        return [(self.feature_names[i], row[i]) for i in top_indices]

    def plot_tfidf_heatmap(self, doc_indices: List[int], n_words: int = 10, title: str = "TF-IDF Heatmap"):
        """Plots a heatmap for a subset of documents and their top terms."""
        # Find top terms across these documents
        important_terms = set()
        for idx in doc_indices:
            top_words = self.get_top_tfidf_words(idx, n=n_words)
            important_terms.update([w for w, s in top_words])
        
        important_terms = sorted(list(important_terms))
        term_indices = [self.vectorizer.vocabulary_[term] for term in important_terms]
        
        subset_matrix = self.tfidf_matrix[doc_indices][:, term_indices].toarray()
        df = pd.DataFrame(subset_matrix, columns=important_terms, index=[f"Doc {i}" for i in doc_indices])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu')
        plt.title(title)
        plt.xlabel('Terms')
        plt.ylabel('Documents')
        plt.show()
