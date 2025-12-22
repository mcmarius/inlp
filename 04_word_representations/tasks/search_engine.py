import pandas as pd
from typing import List, Dict
from representations.bm25 import BM25Representation

class SearchEngine:
    def __init__(self, df: pd.DataFrame, content_column: str = 'content_tokens'):
        """
        df: DataFrame containing the corpus.
        content_column: Column name with tokenized text (list of strings or list of dicts with 'lemma').
        """
        self.df = df
        # Extract lemmas if the tokens are dicts (standard for this project)
        first_doc = df[content_column].iloc[0]
        if isinstance(first_doc[0], dict):
            tokenized_corpus = [[t['lemma'] for t in doc] for doc in df[content_column]]
        else:
            tokenized_corpus = df[content_column].tolist()
            
        self.bm25_model = BM25Representation(tokenized_corpus)

    def search(self, query: str, n: int = 5) -> pd.DataFrame:
        """Searches the corpus and returns top n results."""
        # Simple tokenization for query (can be improved)
        tokenized_query = query.lower().split()
        top_indices = self.bm25_model.get_top_n(tokenized_query, n=n)
        
        results = self.df.iloc[top_indices].copy()
        scores = self.bm25_model.get_scores(tokenized_query)[top_indices]
        results['relevance_score'] = scores
        
        return results[['title', 'date', 'url', 'relevance_score']]

    def display_results(self, query: str, n: int = 5):
        """Displays search results in a nice table."""
        results = self.search(query, n)
        print(f"Search results for: '{query}'")
        if results.empty:
            print("No results found.")
        else:
            from IPython.display import display
            display(results)
