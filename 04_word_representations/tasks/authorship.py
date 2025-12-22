import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List
from utils import get_improved_stopwords

class AuthorshipIdentifier:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.stopwords = get_improved_stopwords()
        # We only want to keep the stopwords as features
        self.vectorizer = CountVectorizer(vocabulary=self.stopwords, lowercase=True)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)

    def extract_features(self, documents: List[str]):
        """Extract relative frequencies of stopwords as stylistic features."""
        dtm = self.vectorizer.fit_transform(documents)
        # Normalize by row sum to get relative frequencies (probabilities)
        features = normalize(dtm, norm='l1', axis=1)
        return features

    def cluster_articles(self, documents: List[str]):
        """Clusters articles based on stopword usage."""
        features = self.extract_features(documents)
        self.labels = self.kmeans.fit_predict(features)
        return self.labels

    def plot_author_clusters(self, documents: List[str], titles: List[str] = None):
        """Visualizes clusters using PCA."""
        features = self.extract_features(documents)
        coords = self.pca.fit_transform(features.toarray())
        
        df_viz = pd.DataFrame({
            'pca_x': coords[:, 0],
            'pca_y': coords[:, 1],
            'cluster': [f"Style Group {l}" for l in self.labels]
        })
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_viz, x='pca_x', y='pca_y', hue='cluster', palette='Set2')
        plt.title('Stylistic Clustering based on Stopword Frequencies')
        plt.xlabel('PCA 1 (Stylistic variance)')
        plt.ylabel('PCA 2 (Stylistic variance)')
        plt.show()

    def get_stylistic_markers(self, n_words: int = 5):
        """Finds the most characteristic stopwords for each cluster."""
        centroids = self.kmeans.cluster_centers_
        markers = []
        for i in range(self.n_clusters):
            # Sort indices of the centroid (which corresponds to stopwords in our vocabulary)
            top_indices = centroids[i].argsort()[::-1][:n_words]
            top_words = [self.stopwords[idx] for idx in top_indices]
            markers.append(top_words)
        return markers
