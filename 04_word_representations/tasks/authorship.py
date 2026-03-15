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
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class AuthorshipIdentifier:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.stopwords = get_improved_stopwords()
        # We only want to keep the stopwords as features
        self.vectorizer = CountVectorizer(vocabulary=self.stopwords, lowercase=True, min_df=2)
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

    def get_detailed_stylistic_markers(self, n_words: int = 5):
        """Returns the top stopwords and their mean frequencies per cluster."""
        # cluster_centers_ shape is (n_clusters, n_features)
        centroids = self.kmeans.cluster_centers_
        feature_names = self.vectorizer.get_feature_names_out()

        cluster_stats = {}

        for i in range(self.n_clusters):
            # Get indices of top N frequencies
            top_indices = centroids[i].argsort()[::-1][:n_words]

            # Create a dict of {word: mean_frequency}
            stats = {
                feature_names[idx]: centroids[i][idx]
                for idx in top_indices
            }
            cluster_stats[f"Cluster {i}"] = stats

        return cluster_stats

    def get_cluster_dataframe(self):
        feature_names = self.vectorizer.get_feature_names_out()
        df = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=feature_names,
            index=[f"Cluster {i}" for i in range(self.n_clusters)]
        )
        # Return the transposed version so words are rows and clusters are columns
        return df.T

    def test_significance(self, documents: List[str]):
        # 1. Get the frequency matrix and labels
        features = self.extract_features(documents).toarray()
        feature_names = self.vectorizer.get_feature_names_out()

        results = []

        # 2. Loop through each stopword and run ANOVA
        for i, word in enumerate(feature_names):
            if np.all(features[:, i] == features[0, i]):
                continue  # Skip words that are the same in every single document
            # Group the frequencies of this specific word by cluster label
            groups = [features[self.labels == j, i] for j in range(self.n_clusters)]

            # F_oneway performs the 1-way ANOVA
            # f_stat, p_val = stats.f_oneway(*groups)
            # Kruskal-Wallis is a non-parametric test.
            # While it can still struggle with tied ranks (all zeros),
            # it is generally more robust for text data than ANOVA
            # (which assumes a normal distribution that word counts rarely follow).
            h_stat, p_val = stats.kruskal(*groups)

            results.append({
                'word': word,
                'h_stat': h_stat,
                'p_value': p_val
            })

        return pd.DataFrame(results).sort_values('p_value')

    def get_pairwise_differences(self, documents: List[str], word: str):
        features = self.extract_features(documents).toarray()
        word_idx = list(self.vectorizer.get_feature_names_out()).index(word)

        # Run Tukey HSD on a specific word
        tukey = pairwise_tukeyhsd(endog=features[:, word_idx],
                                  groups=self.labels,
                                  alpha=0.05)
        return tukey.summary()
