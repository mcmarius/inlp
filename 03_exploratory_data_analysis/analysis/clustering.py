from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from typing import List

class DocumentClusher:
    def __init__(self, n_clusters: int = 5, stopwords: List[str] = None):
        self.n_clusters = n_clusters
        self.stopwords = self._fix_stopwords(stopwords)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2, 
            stop_words=self.stopwords
        )
        self.pca = PCA(n_components=2, random_state=42)

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

    def fit(self, documents: List[str]):
        """Fit clustering model."""
        tfidf = self.vectorizer.fit_transform(documents)
        self.kmeans.fit(tfidf)
        return self.kmeans.labels_

    def get_pca_coords(self, documents: List[str]):
        """Get 2D coordinates for visualization."""
        tfidf = self.vectorizer.transform(documents)
        coords = self.pca.fit_transform(tfidf.toarray())
        return coords

    def get_cluster_descriptors(self, n_words: int = 5) -> List[str]:
        """Get representative words for each cluster based on centroids."""
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()
        
        descriptors = []
        for i in range(self.n_clusters):
            cluster_terms = [terms[ind] for ind in order_centroids[i, :n_words]]
            descriptors.append(", ".join(cluster_terms))
            
        return descriptors
