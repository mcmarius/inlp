import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple
from wordcloud import WordCloud

from utils import get_improved_stopwords

class BoWRepresentation:
    def __init__(self, max_features: int = 1000, stop_words: List[str] = None):
        if stop_words is None:
            stop_words = get_improved_stopwords()
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
        self.feature_names = None
        self.dtm = None

    def fit_transform(self, documents: List[str]):
        """Creates the Document-Term Matrix."""
        self.dtm = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.dtm

    def get_top_n_words(self, n: int = 20) -> List[Tuple[str, int]]:
        """Returns the top n words by total frequency."""
        sum_words = self.dtm.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_top_words(self, n: int = 20, title: str = "Top Words by Frequency (BoW)"):
        """Plots a bar chart of the top n words."""
        top_words = self.get_top_n_words(n)
        df = pd.DataFrame(top_words, columns=['word', 'count'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='count', y='word', data=df, palette='viridis')
        plt.title(title)
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        plt.tight_layout()
        plt.show()

    def plot_wordcloud(self, title: str = "Word Cloud (BoW)"):
        """Generates a word cloud based on frequencies."""
        sum_words = self.dtm.sum(axis=0)
        word_freqs = {word: sum_words[0, idx] for word, idx in self.vectorizer.vocabulary_.items()}
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
