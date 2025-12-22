import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np
from utils import get_improved_stopwords

class BM25Representation:
    def __init__(self, tokenized_corpus: List[List[str]], remove_stopwords: bool = True):
        self.stopwords = get_improved_stopwords() if remove_stopwords else []
        self.corpus = [
            [word for word in doc if word.lower() not in self.stopwords]
            for doc in tokenized_corpus
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def get_scores(self, tokenized_query: List[str]) -> np.ndarray:
        """Returns BM25 scores for all documents in the corpus for a given query."""
        filtered_query = [word for word in tokenized_query if word.lower() not in self.stopwords]
        return self.bm25.get_scores(filtered_query)

    def get_top_n(self, tokenized_query: List[str], n: int = 5) -> List[int]:
        """Returns indices of the top n documents for a query."""
        filtered_query = [word for word in tokenized_query if word.lower() not in self.stopwords]
        scores = self.bm25.get_scores(filtered_query)
        return np.argsort(scores)[::-1][:n].tolist()

    def plot_score_distribution(self, tokenized_query: List[str], title: str = "BM25 Score Distribution"):
        """Plots a histogram of BM25 scores for a query."""
        scores = self.get_scores(tokenized_query)
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, kde=True, color='salmon')
        plt.title(f"{title} for query: {' '.join(tokenized_query)}")
        plt.xlabel('BM25 Score')
        plt.ylabel('Number of Documents')
        plt.show()
