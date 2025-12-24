# Lesson 06: Transformers Applications

In this lesson, we advance from static and basic contextual embeddings to full-fledged **Transformer applications**. We explore Sequence-to-Sequence tasks and advanced topic modeling, building on the concepts from Lesson 04.

## Objectives

1.  **Understand the Shift**: From RNNs to the Transformer architecture (Attention is All You Need).
2.  **Text Summarization**:
    -   **0-shot Extractive**: Using DistilBERT embeddings and graph-based ranking (TextRank) to select key sentences.
    -   **Fine-tuned Abstractive**: Training a Romanian T5 model (`dumitrescustefan/t5-v1_1-base-romanian`) to generate titles from article content.
3.  **Sentence Transformers**: Using Siamese BERT networks for semantic similarity search to find duplicate or related articles.
4.  **BERTopic**: Advanced topic modeling that leverages class-based TF-IDF and embeddings to discover coherent topics in the "General" category.

## Tools

-   **Hugging Face Transformers**: For accessing pre-trained models (DistilBERT, T5).
-   **Sentence-Transformers**: For easy access to semantic similarity models.
-   **BERTopic**: For state-of-the-art topic modeling.
-   **ROUGE Score**: For evaluating summarization quality.

## Setup

1.  Generate the notebook from the JupyText script:
    ```bash
    uv run jupytext --update --to ipynb notebooks/01_transformers_applications.py
    ```
2.  Explore `notebooks/01_transformers_applications.ipynb`.

## Models Used

-   **Embeddings**: `racai/distilbert-base-romanian-cased` (Lightweight, efficient)
-   **Generative**: `dumitrescustefan/t5-v1_1-base-romanian` (For abstractive summarization)
