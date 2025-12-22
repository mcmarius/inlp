import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from representations.tfidf import TFIDFRepresentation

class ArticleRecommender:
    def __init__(self, df: pd.DataFrame, content_column: str = 'lemmatized_content'):
        self.df = df
        self.tfidf_model = TFIDFRepresentation(max_features=5000)
        self.tfidf_matrix = self.tfidf_model.fit_transform(df[content_column].tolist())
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_recommendations(self, article_idx: int, n: int = 5) -> pd.DataFrame:
        """Returns top n similar articles for a given article index."""
        sim_scores = list(enumerate(self.similarity_matrix[article_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the article itself
        sim_scores = sim_scores[1:n+1]
        
        doc_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        results = self.df.iloc[doc_indices].copy()
        results['similarity_score'] = scores
        
        return results[['title', 'date', 'url', 'similarity_score']]

    def plot_similarity_heatmap(self, article_indices: List[int], title: str = "Article Similarity Heatmap"):
        """Plots a heatmap of similarity between a subset of articles."""
        subset_sim = self.similarity_matrix[np.ix_(article_indices, article_indices)]
        
        labels = [f"Art {i}: {self.df.iloc[i]['title'][:30]}..." for i in article_indices]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(subset_sim, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def display_recommendations(self, article_idx: int, n: int = 5):
        """Displays recommendations for an article."""
        article_title = self.df.iloc[article_idx]['title']
        print(f"Recommendations for: '{article_title}'")
        results = self.get_recommendations(article_idx, n)
        from IPython.display import display
        display(results)
