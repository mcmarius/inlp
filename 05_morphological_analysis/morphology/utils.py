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
# Wordnet Helper Functions (using wn package)
# ============================================================================

def get_synset_info(wn_inst, term: str, synset_index: int = 0) -> Optional[Dict]:
    """
    Get basic information about a synset.
    
    Args:
        wn_inst: Wordnet instance (e.g., wn.Wordnet('omw-ro:2.0'))
        term: The word to look up
        synset_index: Which synset to use (default: 0 = first)
    
    Returns:
        Dict with synset info or None if not found
    """
    words = wn_inst.words(term)
    if not words:
        return None
        
    synsets = []
    for w in words:
        synsets.extend(w.synsets())
        
    if not synsets or synset_index >= len(synsets):
        return None
    
    synset = synsets[synset_index]
    
    return {
        "term": term,
        "synset_id": synset.id,
        "literals": synset.lemmas(),
        "definition": synset.definition(),
    }


def get_hypernym_chain(wn_inst, term: str, synset_index: int = 0, max_depth: int = 5) -> List[str]:
    """
    Get the hypernym chain for a term (from specific to general).
    
    Args:
        wn_inst: Wordnet instance
        term: The word to look up
        synset_index: Which synset to use
        max_depth: Maximum number of hypernyms to retrieve
    
    Returns:
        List of hypernym literals
    """
    words = wn_inst.words(term)
    if not words:
        return []
        
    synsets = []
    for w in words:
        synsets.extend(w.synsets())
        
    if not synsets or synset_index >= len(synsets):
        return []
    
    synset = synsets[synset_index]
    
    chain = []
    current_synset = synset
    
    for _ in range(max_depth):
        try:
            hypernyms = current_synset.hypernyms()
        except:
            break
            
        if not hypernyms:
            break
            
        # Take the first hypernym
        current_synset = hypernyms[0]
        lemmas = current_synset.lemmas()
        if lemmas:
            chain.append(lemmas[0])
        else:
            chain.append(f"<{current_synset.id}>")
        
    return chain


def get_all_relations(wn_inst, term: str, synset_index: int = 0) -> Dict[str, List[str]]:
    """
    Get all relations for a synset.
    (Note: wn handles relations differently, we will group them by relation type)
    
    Args:
        wn_inst: Wordnet instance
        term: The word to look up
        synset_index: Which synset to use
    
    Returns:
        Dict with 'outbound' and 'inbound' relations (for compatibility, though wn relations are bidirectional in API)
    """
    words = wn_inst.words(term)
    if not words:
        return {"outbound": [], "inbound": []}
        
    synsets = []
    for w in words:
        synsets.extend(w.synsets())
        
    if not synsets or synset_index >= len(synsets):
        return {"outbound": [], "inbound": []}
    
    synset = synsets[synset_index]
    
    # We map 'wn' relations to the old list format for compatibility
    outbound = []
    for rel_type, related_synsets in synset.relations().items():
        for rs in related_synsets:
            # We skip INFERRED nodes as they don't have lemmas
            if rs.id == '*INFERRED*':
                continue
            lemmas = rs.lemmas()
            if lemmas:
                outbound.append({"relation": rel_type, "target": lemmas[0]})
            else:
                outbound.append({"relation": rel_type, "target": rs.id})
                
    # `wn` doesn't strictly separate inbound/outbound at the relation dict level the same way,
    # but we can just return empty inbound for compatibility, or populate it if needed.
    return {"outbound": outbound, "inbound": []}


def find_semantic_path(wn_inst, term1: str, term2: str) -> Optional[List[str]]:
    """
    Find the shortest semantic path between two terms.
    
    Args:
        wn_inst: Wordnet instance
        term1: First term
        term2: Second term
    
    Returns:
        List of synset literals in the path, or None if no path found
    """
    words1 = wn_inst.words(term1)
    words2 = wn_inst.words(term2)
    
    if not words1 or not words2:
        return None
        
    synsets1 = words1[0].synsets()
    synsets2 = words2[0].synsets()
    
    if not synsets1 or not synsets2:
        return None
        
    synset1 = synsets1[0]
    synset2 = synsets2[0]
    
    try:
        path = synset1.shortest_path(synset2)
        if not path:
            return None
        return [s.lemmas()[0] for s in path if s.lemmas()]
    except:
        return None


def find_common_ancestor(wn_inst, term1: str, term2: str) -> Optional[str]:
    """
    Find the lowest common hypernym ancestor of two terms.
    
    Args:
        wn_inst: Wordnet instance
        term1: First term
        term2: Second term
    
    Returns:
        The common ancestor literal, or None if not found
    """
    words1 = wn_inst.words(term1)
    words2 = wn_inst.words(term2)
    
    if not words1 or not words2:
        return None
        
    synsets1 = words1[0].synsets()
    synsets2 = words2[0].synsets()
    
    if not synsets1 or not synsets2:
        return None
        
    synset1 = synsets1[0]
    synset2 = synsets2[0]
    
    try:
        ancestors = synset1.lowest_common_hypernyms(synset2)
        if not ancestors:
            return None
            
        for a in ancestors:
            lemmas = a.lemmas()
            if lemmas:
                return lemmas[0]
        return None
    except:
        return None


def extract_synonyms(wn_inst, terms: List[str]) -> List[Tuple[str, str]]:
    """
    Extract synonym pairs from a list of terms.
    
    Args:
        wn_inst: Wordnet instance
        terms: List of terms to extract synonyms from
    
    Returns:
        List of (synonym1, synonym2) tuples
    """
    synonyms = []
    
    for term in terms:
        words = wn_inst.words(term)
        if not words:
            continue
            
        synsets = []
        for w in words:
            synsets.extend(w.synsets())
            
        for synset in synsets[:1]:  # Just first synset
            literals = synset.lemmas()
            
            # Create pairs from literals in the same synset
            for i in range(len(literals)):
                for j in range(i + 1, len(literals)):
                    synonyms.append((literals[i], literals[j]))
    
    return list(set(synonyms))  # Remove duplicates


def extract_antonyms(wn_inst, terms: List[str]) -> List[Tuple[str, str]]:
    """
    Extract antonym pairs from a list of terms.
    
    Args:
        wn_inst: Wordnet instance
        terms: List of terms to extract antonyms from
    
    Returns:
        List of (word, antonym) tuples
    """
    antonyms = []
    
    for term in terms:
        words = wn_inst.words(term)
        if not words:
            continue
            
        synsets = []
        for w in words:
            synsets.extend(w.synsets())
            
        for synset in synsets[:2]:  # First 2 synsets
            relations = synset.relations()
            
            # Check for antonym relations (in wn, it's often 'antonym' or similar)
            # In omw, near_antonym might be mapped differently or just not present
            # We'll check 'antonym' and 'near_antonym'
            antonym_synsets = []
            for rel_type in ['antonym', 'near_antonym']:
                if rel_type in relations:
                    antonym_synsets.extend(relations[rel_type])
            
            for synset_antonym in antonym_synsets:
                # Generate cartesian product of literals
                synset_literals = synset.lemmas()
                antonym_literals = synset_antonym.lemmas()
                
                for lit1, lit2 in itertools.product(synset_literals, antonym_literals):
                    antonyms.append((lit1, lit2))
    
    return list(set(antonyms))  # Remove duplicates
