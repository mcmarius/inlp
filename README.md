# Introduction to Natural Language Processing

## Outline

We use [JupyText](https://jupytext.readthedocs.io/en/latest/) to write our notebooks in `.py` files for better version control, which are then converted to `.ipynb` files.


### 1. Building a dataset

In [`01_data_collection`](01_data_collection) we will build a dataset from scratch. This directory will contain scripts for more datasets in the future.

#### ANPC Press Releases

For now, we are using web scraping to extract press release articles from the [ANPC website](https://anpc.ro/category/comunicate-de-presa/).

The ANPC website uses a custom script to hydrate the page, which requires a JS runtime to render the actual page. By the way, in the hydration script, there is `var maioneza`; can't make this up. In early 2025 when I made a simple shell script with `curl`, the `maioneza` was not there.

We use [Playwright](https://playwright.dev/) to render each page and extract the HTML.


### 2. Data preprocessing

In [`02_data_preprocessing`](02_data_preprocessing) we will preprocess the dataset.

#### ANPC Press Releases

This dataset is fairly clean. We normalize diacritics, normalize dates, and tokenize the text.


### 3. Exploratory data analysis

In [`03_exploratory_data_analysis`](03_exploratory_data_analysis) we explore the dataset through temporal analysis, topic modeling, and clustering.


### 4. Word representations

In [`04_word_representations`](04_word_representations) we explore different word representations.

#### Sparse representations
We implement and visualize:
- Bag of Words (BoW)
- TF-IDF

We apply these to the ANPC dataset for:
- Authorship Identification
- Seasonal Analysis

#### Dense representations
- static: word2vec, glove, fasttext
- contextual: transformers


### 5. Morphological analysis, WordNet

In [`05_morphological_analysis`](05_morphological_analysis) we explore the morphological and semantic structure of Romanian text.

#### POS Tagging and NER
- Part-of-speech distribution analysis
- Named entity recognition for organizations, monetary values, and locations
- Verb analysis revealing bureaucratic language patterns

#### Dependency Parsing
- Syntactic structure visualization
- Sentence complexity analysis

#### WordNet
- Semantic relations using RoWordNet
- Hypernyms, hyponyms, and synonyms for consumer protection terms


### 6. Transformers Applications

In [`06_transformers`](06_transformers) we explore transformer-based models and their practical applications.

#### Text Summarization
- **Extractive**: Using DistilBERT with TextRank/PageRank for sentence selection
- **Abstractive**: Fine-tuning T5 for title generation (with documented limitations on small datasets)

#### Semantic Search
- Building Sentence Transformers (SBERT) for semantic similarity
- Duplicate detection and related article recommendation

#### Topic Modeling with BERTopic
- Advanced topic modeling using transformer embeddings
- Stopword-filtered topics for meaningful domain term discovery
- Temporal topic analysis to track themes over time


### 7. LLMs


### 8. Feature importance, explainability, interpretability

In [`08_explainability`](08_explainability) we explore how and why NLP models make predictions.

#### Traditional ML Feature Importance
- Logistic Regression coefficients for direct interpretation
- Random Forest Gini importance
- Comparative analysis across methods

#### LIME (Local Interpretable Model-agnostic Explanations)
- Local explanations for individual predictions
- Model-agnostic approach working with any classifier
- Text-specific perturbation strategies

#### SHAP (SHapley Additive exPlanations)
- Game-theoretic feature attribution
- Waterfall, summary, and force plot visualizations
- Consistent and locally accurate explanations

#### Transformer Attention Visualization
- Self-attention mechanism analysis
- Multi-head and multi-layer attention patterns
- Identifying important tokens based on attention scores

#### Comparative Analysis
- Cross-method validation of feature importance
- Understanding when different methods agree/disagree
- Practical guidelines for choosing explainability techniques
