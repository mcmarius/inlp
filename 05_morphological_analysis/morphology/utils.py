"""Utility functions for morphological analysis and WordNet exploration."""

import spacy
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
import itertools


def load_spacy_model(model_name: str = "ro_core_news_sm"):
    """Load SpaCy Romanian model."""
    return spacy.load(model_name)


def extract_pos_stats(docs: List) -> pd.DataFrame:
    """Extract POS tag statistics from SpaCy docs."""
    pos_counts = Counter()
    
    for doc in docs:
        for token in doc:
            pos_counts[token.pos_] += 1
    
    total = sum(pos_counts.values())
    data = [
        {"POS": pos, "Count": count, "Percentage": count / total * 100}
        for pos, count in pos_counts.most_common()
    ]
    
    return pd.DataFrame(data)


def extract_entities(docs: List) -> pd.DataFrame:
    """Extract named entities from SpaCy docs."""
    entity_counts = Counter()
    
    for doc in docs:
        for ent in doc.ents:
            entity_counts[(ent.text, ent.label_)] += 1
    
    data = [
        {"Entity": text, "Label": label, "Count": count}
        for (text, label), count in entity_counts.most_common(50)
    ]
    
    return pd.DataFrame(data)


def extract_top_verbs(docs: List, top_n: int = 20) -> pd.DataFrame:
    """Extract most common verbs from SpaCy docs."""
    verb_counts = Counter()
    
    for doc in docs:
        for token in doc:
            if token.pos_ == "VERB":
                verb_counts[token.lemma_.lower()] += 1
    
    data = [
        {"Verb": verb, "Count": count}
        for verb, count in verb_counts.most_common(top_n)
    ]
    
    return pd.DataFrame(data)


def get_dependency_depth(doc) -> int:
    """Calculate the maximum dependency tree depth for a sentence."""
    def depth(token):
        if not list(token.children):
            return 1
        return 1 + max(depth(child) for child in token.children)
    
    return max(depth(sent.root) for sent in doc.sents)


def analyze_sentence_complexity(docs: List) -> Dict[str, float]:
    """Analyze sentence complexity metrics."""
    depths = []
    lengths = []
    
    for doc in docs:
        for sent in doc.sents:
            depths.append(get_dependency_depth(spacy.tokens.Doc(doc.vocab, words=[t.text for t in sent])))
            lengths.append(len(sent))
    
    return {
        "avg_depth": sum(depths) / len(depths) if depths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_depth": max(depths) if depths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


# ============================================================================
# RoWordNet Helper Functions
# ============================================================================

def get_synset_info(wn, term: str, synset_index: int = 0) -> Optional[Dict]:
    """
    Get basic information about a synset.
    
    Args:
        wn: RoWordNet instance
        term: The word to look up
        synset_index: Which synset to use (default: 0 = first)
    
    Returns:
        Dict with synset info or None if not found
    """
    synsets = wn.synsets(term)
    if not synsets or synset_index >= len(synsets):
        return None
    
    synset_id = synsets[synset_index]
    synset = wn.synset(synset_id)
    
    return {
        "term": term,
        "synset_id": synset_id,
        "literals": list(synset.literals),
        "definition": synset.definition,
    }


def get_hypernym_chain(wn, term: str, synset_index: int = 0, max_depth: int = 5) -> List[str]:
    """
    Get the hypernym chain for a term (from specific to general).
    
    Args:
        wn: RoWordNet instance
        term: The word to look up
        synset_index: Which synset to use
        max_depth: Maximum number of hypernyms to retrieve
    
    Returns:
        List of hypernym literals
    """
    synsets = wn.synsets(term)
    if not synsets or synset_index >= len(synsets):
        return []
    
    synset_id = synsets[synset_index]
    hypernyms = wn.synset_to_hypernym_root(synset_id)
    
    return [wn.synset(h).literals[0] for h in hypernyms[:max_depth]]


def get_all_relations(wn, term: str, synset_index: int = 0) -> Dict[str, List[str]]:
    """
    Get all relations (inbound and outbound) for a synset.
    
    Args:
        wn: RoWordNet instance
        term: The word to look up
        synset_index: Which synset to use
    
    Returns:
        Dict with 'outbound' and 'inbound' relations
    """
    synsets = wn.synsets(term)
    if not synsets or synset_index >= len(synsets):
        return {"outbound": [], "inbound": []}
    
    synset_id = synsets[synset_index]
    
    outbound = [
        {"relation": rel, "target": wn.synset(target_id).literals[0]}
        for target_id, rel in wn.outbound_relations(synset_id)
    ]
    
    inbound = [
        {"relation": rel, "source": wn.synset(source_id).literals[0]}
        for source_id, rel in wn.inbound_relations(synset_id)
    ]
    
    return {"outbound": outbound, "inbound": inbound}


def find_semantic_path(wn, term1: str, term2: str) -> Optional[List[str]]:
    """
    Find the shortest semantic path between two terms.
    
    Args:
        wn: RoWordNet instance
        term1: First term
        term2: Second term
    
    Returns:
        List of synset literals in the path, or None if no path found
    """
    synsets1 = wn.synsets(term1)
    synsets2 = wn.synsets(term2)
    
    if not synsets1 or not synsets2:
        return None
    
    synset1_id = synsets1[0]
    synset2_id = synsets2[0]
    
    path = wn.shortest_path(synset1_id, synset2_id)
    
    if not path:
        return None
    
    return [wn.synset(sid).literals[0] for sid in path]


def find_common_ancestor(wn, term1: str, term2: str) -> Optional[str]:
    """
    Find the lowest common hypernym ancestor of two terms.
    
    Args:
        wn: RoWordNet instance
        term1: First term
        term2: Second term
    
    Returns:
        The common ancestor literal, or None if not found
    """
    synsets1 = wn.synsets(term1)
    synsets2 = wn.synsets(term2)
    
    if not synsets1 or not synsets2:
        return None
    
    synset1_id = synsets1[0]
    synset2_id = synsets2[0]
    
    ancestor_id = wn.lowest_hypernym_common_ancestor(synset1_id, synset2_id)
    
    if not ancestor_id:
        return None
    
    return wn.synset(ancestor_id).literals[0]


def extract_synonyms(wn, terms: List[str]) -> List[Tuple[str, str]]:
    """
    Extract synonym pairs from a list of terms.
    
    Args:
        wn: RoWordNet instance
        terms: List of terms to extract synonyms from
    
    Returns:
        List of (synonym1, synonym2) tuples
    """
    synonyms = []
    
    for term in terms:
        synsets_id = wn.synsets(term)
        for synset_id in synsets_id[:1]:  # Just first synset
            synset = wn.synset(synset_id)
            literals = list(synset.literals)
            
            # Create pairs from literals in the same synset
            for i in range(len(literals)):
                for j in range(i + 1, len(literals)):
                    synonyms.append((literals[i], literals[j]))
    
    return list(set(synonyms))  # Remove duplicates


def extract_antonyms(wn, terms: List[str]) -> List[Tuple[str, str]]:
    """
    Extract antonym pairs from a list of terms.
    
    Args:
        wn: RoWordNet instance
        terms: List of terms to extract antonyms from
    
    Returns:
        List of (word, antonym) tuples
    """
    antonyms = []
    
    for term in terms:
        synsets_id = wn.synsets(term)
        for synset_id in synsets_id[:2]:  # First 2 synsets
            synset = wn.synset(synset_id)
            
            # Extract antonym relations
            synset_outbound = wn.outbound_relations(synset.id)
            synset_antonyms_id = [
                synset_tuple[0] for synset_tuple in synset_outbound
                if synset_tuple[1] == 'near_antonym'
            ]
            
            for synset_antonym_id in synset_antonyms_id:
                synset_antonym = wn.synset(synset_antonym_id)
                
                # Generate cartesian product of literals
                synset_literals = list(synset.literals)
                antonym_literals = list(synset_antonym.literals)
                
                for lit1, lit2 in itertools.product(synset_literals, antonym_literals):
                    antonyms.append((lit1, lit2))
    
    return list(set(antonyms))  # Remove duplicates
