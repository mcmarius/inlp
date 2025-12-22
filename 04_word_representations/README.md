# Module 04: Word Representations (Sparse)

In this module, we explore sparse word representations using the ANPC press releases dataset. We focus on refining these representations with domain-specific stopwords and applying them to advanced NLP tasks.

## Objectives

- Understand and implement **Bag of Words (BoW)**, **TF-IDF**, and **BM25**.
- Learn the importance of **stopword removal** for meaningful representations.
- Apply representations to advanced tasks:
    - **Authorship Identification**: Clustering articles based on stylistic markers (function words/stopwords).
    - **Seasonal Issue Analysis**: Identifying trending issues across different seasons.
- Practical exercises on **Search Engines**, **Recommender Systems**, and **Keyword Extraction**.

## Contents

- `representations/`: Core implementation of sparse representations.
    - `bow.py`: Bag of Words with improved stopwords.
    - `tfidf.py`: TF-IDF with refined feature extraction.
    - `bm25.py`: BM25 with intelligent query filtering.
- `tasks/`: Practical applications.
    - `authorship.py`: Stylistic clustering based on stopword frequencies.
    - `seasonal_analysis.py`: Trending issues by season (Winter, Spring, Summer, Autumn).
    - `search_engine.py` (Exercise): BM25-based document retrieval.
    - `recommender.py` (Exercise): Content-based similarity.
    - `keyword_extraction.py` (Exercise): Automated tagging.
- `notebooks/`: Interactive walkthroughs.
    - `04_sparse_representations.py`: Main educational notebook (JupyText).
- `utils.py`: Utility functions for improved stopwords and path handling.

## How to use

1. Generate the notebook from the JupyText script:
   ```bash
   uv run jupytext --update --to ipynb notebooks/04_sparse_representations.py
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
