"""Tests for morphology utilities."""

import pytest
import spacy
from morphology.utils import (
    load_spacy_model,
    extract_pos_stats,
    extract_entities,
    extract_top_verbs,
)


def test_load_spacy_model():
    """Test that SpaCy model loads correctly."""
    nlp = load_spacy_model("ro_core_news_sm")
    assert nlp is not None
    assert nlp.lang == "ro"


def test_extract_pos_stats():
    """Test POS statistics extraction."""
    nlp = load_spacy_model("ro_core_news_sm")
    docs = [nlp("ANPC a aplicat amenzi de 1000 lei.")]
    
    pos_df = extract_pos_stats(docs)
    
    assert not pos_df.empty
    assert "POS" in pos_df.columns
    assert "Count" in pos_df.columns
    assert "Percentage" in pos_df.columns
    assert pos_df["Percentage"].sum() == pytest.approx(100.0, rel=1e-2)


def test_extract_entities():
    """Test entity extraction."""
    nlp = load_spacy_model("ro_core_news_sm")
    docs = [nlp("ANPC a aplicat amenzi în București.")]
    
    entities_df = extract_entities(docs)
    
    assert "Entity" in entities_df.columns
    assert "Label" in entities_df.columns
    assert "Count" in entities_df.columns


def test_extract_top_verbs():
    """Test verb extraction."""
    nlp = load_spacy_model("ro_core_news_sm")
    docs = [nlp("ANPC verifică și aplică sancțiuni.")]
    
    verbs_df = extract_top_verbs(docs, top_n=10)
    
    assert "Verb" in verbs_df.columns
    assert "Count" in verbs_df.columns
