import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from utils import get_improved_stopwords

class SeasonalAnalyzer:
    def __init__(self, df: pd.DataFrame, date_column: str = 'date_iso'):
        self.df = df.copy()
        self.df['date_dt'] = pd.to_datetime(self.df[date_column])
        self.df['season'] = self.df['date_dt'].dt.month.map(self._month_to_season)
        self.stopwords = get_improved_stopwords()

    def _month_to_season(self, month: int) -> str:
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        if month in [9, 10, 11]: return 'Autumn'
        return 'Unknown'

    def get_seasonal_keywords(self, n_words: int = 10, max_season_overlap: int = 3) -> Dict[str, Dict[str, float]]:
        """
        Extracts top bigrams for each season using a TF-IDF approach.
        Applies post-processing to remove bigrams that appear in more than 'max_season_overlap' seasons.
        """
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        # Create 'super-documents' for each season
        season_texts = []
        valid_seasons = []
        for season in seasons:
            text = " ".join(self.df[self.df['season'] == season]['lemmatized_content'].tolist())
            if text:
                season_texts.append(text)
                valid_seasons.append(season)

        if not season_texts:
            return {s: {} for s in seasons}

        # Use bigrams only as requested by user
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words=self.stopwords,
            ngram_range=(2, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(season_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Post-processing: Identify how many seasons each bigram appears in
        # We look at the count of non-zero entries in each column of the TF-IDF matrix
        appearance_counts = (tfidf_matrix > 0).sum(axis=0).A1
        
        results = {s: {} for s in seasons}
        for i, season in enumerate(valid_seasons):
            row = tfidf_matrix[i].toarray().flatten()
            
            # Filter terms that appear in too many seasons
            # and terms with 0 score in current season
            mask = (appearance_counts <= max_season_overlap) & (row > 0)
            
            filtered_indices = np.where(mask)[0]
            filtered_scores = row[filtered_indices]
            
            # Sort and take top n
            top_indices = filtered_indices[filtered_scores.argsort()[::-1][:n_words]]
            results[season] = {feature_names[idx]: float(row[idx]) for idx in top_indices}
            
        return results

    def plot_seasonal_trends(self, n_words: int = 10):
        """Plots bar charts of top keywords per season."""
        keywords_dict = self.get_seasonal_keywords(n_words=n_words)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        for i, season in enumerate(seasons):
            season_data = keywords_dict[season]
            if not season_data:
                axes[i].text(0.5, 0.5, 'No Data', ha='center')
                axes[i].set_title(season)
                continue
                
            words = list(season_data.keys())
            scores = list(season_data.values())
            
            sns.barplot(
                x=scores, 
                y=words, 
                ax=axes[i], 
                palette='viridis', 
                hue=words, 
                legend=False
            )
            axes[i].set_title(f"Unique Issues in {season}")
            axes[i].set_xlabel("TF-IDF Score (Season Specificity)")
            
        plt.tight_layout()
        plt.show()
