import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import List, Dict
from representations.tfidf import TFIDFRepresentation

class KeywordExtractor:
    def __init__(self, df: pd.DataFrame, content_column: str = 'lemmatized_content'):
        self.df = df
        self.tfidf_model = TFIDFRepresentation(max_features=5000)
        self.tfidf_model.fit_transform(df[content_column].tolist())

    def get_keywords(self, article_idx: int, n: int = 10) -> List[str]:
        """Extract top n keywords for an article."""
        top_words = self.tfidf_model.get_top_tfidf_words(article_idx, n=n)
        return [word for word, score in top_words]

    def plot_keywords_wordcloud(self, article_idx: int, title: str = None):
        """Plots a word cloud for an article's keywords."""
        top_words = self.tfidf_model.get_top_tfidf_words(article_idx, n=30)
        word_freqs = {word: score for word, score in top_words}
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
        
        if title is None:
            title = f"Keywords for: {self.df.iloc[article_idx]['title'][:50]}..."
            
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()

    def add_keywords_to_df(self, n: int = 5):
        """Adds a column with keywords to the DataFrame."""
        keywords_list = []
        for i in range(len(self.df)):
            keywords_list.append(", ".join(self.get_keywords(i, n=n)))
        self.df['keywords'] = keywords_list
        return self.df
