# Module 04: Word Representations (Sparse & Dense)

In this module, we explore sparse word representations using the ANPC press releases dataset. We focus on refining these representations with domain-specific stopwords and applying them to advanced NLP tasks.

## Objectives

- Understand and implement **Bag of Words (BoW)**, **TF-IDF**, and **BM25**.
- Explore **Static Dense Representations**: **Word2Vec**, **GloVe**, and **FastText**.
- Discover **Contextual Word Representations**: **BERT** and **DistilBERT** for Romanian.
- Learn the importance of **stopword removal** for meaningful representations.
- Apply representations to advanced tasks:
    - **Authorship Identification**: Clustering articles based on stylistic markers.
    - **Seasonal Issue Analysis**: Identifying trending issues across different seasons.
    - **Temporal Semantic Shifts**: Analyzing how word meanings evolve over time.
    - **Polysemy Analysis**: Visualizing how the same word changes meaning in context.
- Practical exercises on **Search Engines**, **Recommender Systems**, **Semantic Analogies**, and **Contextual Extraction**.

## Contents

- `representations/`: Core implementation of word representations.
    - `bow.py`: Bag of Words with improved stopwords.
    - `tfidf.py`: TF-IDF with refined feature extraction.
    - `bm25.py`: BM25 with intelligent query filtering.
    - `embeddings.py`: Wrapper for dense representations (Word2Vec, FastText).
- `tasks/`: Practical applications.
    - `authorship.py`: Stylistic clustering based on stopword frequencies.
    - `seasonal_analysis.py`: Trending issues by season.
- `notebooks/`: Interactive walkthroughs.
    - `04_sparse_representations.py`: Sparse representations (BoW, TF-IDF, BM25).
    - `05_dense_representations.py`: Static dense representations (Word2Vec, FastText, Visualizations).
    - `06_contextual_representations.py`: Contextual representations (DistilBERT, Polysemy, UMAP).
- `utils.py`: Utility functions for improved stopwords and path handling.

## How to use

1. Generate the notebook from the JupyText script:
   ```bash
   uv run jupytext --update --to ipynb notebooks/04_sparse_representations.py
   uv run jupytext --update --to ipynb notebooks/05_dense_representations.py
   uv run jupytext --update --to ipynb notebooks/06_contextual_representations.py
   ```
2. Explore `notebooks/04_sparse_representations.ipynb` to see the results.
3. Run the unit tests:
   ```bash
   uv run pytest tests/
   ```

## Visualizations

- **Stylistic Clusters**: Visualization of how articles group together based on writing style.
- **Seasonal Trends**: Comparison of top keywords and issues reported in different seasons.
- **Improved Heatmaps & Clouds**: Filtered visualizations that reveal actual content insights.
- **Contextual Clusters**: Visualization of polysemous words (e.g., "bancă") in different contexts.

> [!NOTE]
> **GPU Usage**: This project is not using a GPU (might not be available e.g. in CI). If you have a GPU, the `transformers` library will automatically detect it (ensure `cuda` is available), but all labs are designed to work on a standard laptop CPU using lightweight models like DistilBERT.
