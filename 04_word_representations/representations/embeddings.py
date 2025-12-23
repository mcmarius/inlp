import numpy as np
from gensim.models import Word2Vec, FastText
import pandas as pd
from typing import List, Union, Dict, Any
import os

class EmbeddingWrapper:
    """
    A wrapper for Gensim's Word2Vec and FastText models to provide a consistent interface.
    """
    def __init__(self, model_type: str = "word2vec", **kwargs):
        self.model_type = model_type.lower()
        self.model = None
        self.kwargs = kwargs
        if self.model_type not in ["word2vec", "fasttext"]:
            raise ValueError("model_type must be either 'word2vec' or 'fasttext'")

    def train(self, sentences: List[List[str]]):
        """
        Train the model on a list of tokenized sentences.
        """
        if self.model_type == "word2vec":
            self.model = Word2Vec(sentences, **self.kwargs)
        elif self.model_type == "fasttext":
            self.model = FastText(sentences, **self.kwargs)
        
    def get_vector(self, word: str) -> np.ndarray:
        """
        Get the vector representation of a word.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if self.model_type == "word2vec":
            if word in self.model.wv:
                return self.model.wv[word]
            else:
                return np.zeros(self.model.vector_size)
        elif self.model_type == "fasttext":
            # FastText can handle OOV words
            return self.model.wv[word]

    def most_similar(self, word: str, topn: int = 10) -> List[Dict[str, Union[str, float]]]:
        """
        Find the most similar words.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if self.model_type == "word2vec" and word not in self.model.wv:
            return []
            
        similar_words = self.model.wv.most_similar(word, topn=topn)
        return [{"word": w, "similarity": float(s)} for w, s in similar_words]

    def analogy(self, a: str, b: str, c: str, topn: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Solve analogy: a is to b as c is to ?
        b - a + c = ?
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            similar_words = self.model.wv.most_similar(positive=[b, c], negative=[a], topn=topn)
            return [{"word": w, "similarity": float(s)} for w, s in similar_words]
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return []

    def save(self, path: str):
        """
        Save the model to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save(path)

    @classmethod
    def load(cls, path: str, model_type: str):
        """
        Load a model from disk.
        """
        instance = cls(model_type=model_type)
        if model_type.lower() == "word2vec":
            instance.model = Word2Vec.load(path)
        elif model_type.lower() == "fasttext":
            instance.model = FastText.load(path)
        return instance

def get_word_vectors(model, words: List[str]) -> np.ndarray:
    """
    Get vectors for a list of words.
    """
    vectors = []
    for word in words:
        vectors.append(model.get_vector(word))
    return np.array(vectors)
