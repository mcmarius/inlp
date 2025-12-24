# Lesson 05: Morphological Analysis and WordNet

In this lesson, we explore the structural and semantic layers of the Romanian language using the ANPC press releases dataset. We focus on:

- **POS Tagging (Part-of-Speech)**: Identifying nouns, verbs, adjectives, etc., to understand stylistic patterns.
- **NER (Named Entity Recognition)**: Extracting organizations, locations, and people mentioned in the press releases.
- **Dependency Parsing**: Analyzing the syntactic structure of sentences.
- **WordNet**: Using `rowordnet` to explore semantic relations like hypernymy and synonymy.

## Objectives

1.  **Linguistic Insights**: What are the most common entities? How has the language of ANPC evolved?
2.  **Structural Patterns**: Understanding how sentences are constructed.
3.  **Semantic Enrichment**: Mapping words to concepts using WordNet.

## Tools

-   **SpaCy**: For the Romanian NLP pipeline (POS, NER, Parsing).
-   **RoWordNet**: The Romanian WordNet implementation.
-   **Seaborn/Matplotlib**: For visualizing linguistics statistics.

## Dataset

We continue using the **ANPC Press Releases** dataset, which provides a rich source of official, yet often descriptive, Romanian text.

## Setup

1. Generate the notebook from the JupyText script:
   ```bash
   uv run jupytext --update --to ipynb notebooks/01_morphology_and_wordnet.py
   ```
2. Explore `notebooks/01_morphology_and_wordnet.ipynb` to see the results.
3. Run the unit tests:
   ```bash
   uv run pytest tests/
   ```
