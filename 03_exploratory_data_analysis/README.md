# Lesson 03: Exploratory Data Analysis & Topic Modeling

This lesson focuses on exploring the ANPC press releases dataset and extracting insights through temporal analysis, topic modeling, and clustering.

## Objectives

1.  **Temporal EDA**: Analyze the distribution of press releases over time.
2.  **Date Interval Extraction**: Extract and normalize date intervals mentioned in the text.
3.  **Topic Modeling**: Categorize articles using LDA, NMF, and BM25.
4.  **Clustering**: Group related press releases based on semantic similarity.

## Setup

```bash
# Sync dependencies
uv sync --all-extras

# Generate the notebook
uv run jupytext --update --to ipynb notebooks/01_eda_anpc.py
```

## Structure

- `analysis/`: Core logic for date extraction, topic modeling, and clustering.
- `notebooks/`: Jupytext notebooks for interactive exploration.
- `tests/`: Automated tests for the analysis logic.

## TODO
- fix duplicate words in wordclouds
- determine topic names for clusters