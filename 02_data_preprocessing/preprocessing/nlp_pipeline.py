import stanza
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
from typing import List, Dict

# Force download if missing (handled in setup but good to be safe)
# stanza.download('ro')

class RomanianNLP:
    def __init__(self):
        # Initialize Stanza
        self.stanza_pipe = stanza.Pipeline(lang='ro', processors='tokenize,pos,lemma')
        
        self.spacy_pipe = spacy.load('ro_core_news_sm')
            
        # Initialize NLTK
        self.stemmer = SnowballStemmer("romanian")
        self.ro_stopwords = set(nltk_stopwords.words('romanian'))

    def process_with_stanza(self, text: str) -> List[Dict]:
        """Process text using Stanza and return list of tokens with lemmas."""
        doc = self.stanza_pipe(text)
        results = []
        for sentence in doc.sentences:
            for word in sentence.words:
                results.append({
                    "text": word.text,
                    "lemma": word.lemma,
                    "pos": word.upos
                })
        return results

    def process_with_spacy(self, text: str) -> List[Dict]:
        """Process text using SpaCy and return list of tokens with lemmas."""
        if not self.spacy_pipe:
            return []
        doc = self.spacy_pipe(text)
        return [{"text": token.text, "lemma": token.lemma_, "pos": token.pos_} for token in doc]

    def process_with_nltk(self, text: str) -> List[Dict]:
        """Process text using NLTK: tokenization and stemming."""
        tokens = nltk.word_tokenize(text)
        return [{"text": t, "stem": self.stemmer.stem(t)} for t in tokens]

    def compare_stemming_vs_lemmatization(self, text: str) -> List[Dict]:
        """Show the difference between stemming and lemmatization for Romanian."""
        # Stanza Lemmatization
        doc = self.stanza_pipe(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        
        # NLTK Stemming
        tokens = nltk.word_tokenize(text)
        stems = [self.stemmer.stem(t) for t in tokens]
        
        results = []
        for t, s, l in zip(tokens, stems, lemmas):
            results.append({
                "text": t,
                "stem": s,
                "lemma": l
            })
        return results

def get_romanian_stopwords() -> List[str]:
    """Return a list of Romanian stopwords from NLTK."""
    return sorted(list(nltk_stopwords.words('romanian')))
