# Introduction to Natural Language Processing

## Outline

We use [JupyText](https://jupytext.readthedocs.io/en/latest/) to write our notebooks in `.py` files for better version control, which are then converted to `.ipynb` files.

### Building a dataset

In [`01_data_collection`](01_data_collection) we will build a dataset from scratch. This directory will contain scripts for more datasets in the future.

#### ANPC Press Releases

For now, we are using web scraping to extract press release articles from the [ANPC website](https://anpc.ro/category/comunicate-de-presa/).

The ANPC website uses the Brizy page builder, which requires a JS runtime to render the page (by the way, in the hydration script, there is `var maioneza`; can't make this up).

We use [Playwright](https://playwright.dev/) to render each page and extract the HTML.

### Data preprocessing

In [`02_data_preprocessing`](02_data_preprocessing) we will preprocess the dataset.

#### ANPC Press Releases

This dataset is fairly clean. We normalize diacritics, normalize dates, and tokenize the text.

### Exploratory data analysis

In [`03_exploratory_data_analysis`](03_exploratory_data_analysis) we explore the dataset through temporal analysis, topic modeling, and clustering.

### Word representations

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

### Morphological analysis, WordNet
### Transformers
### LLMs
### Feature importance, explainability
