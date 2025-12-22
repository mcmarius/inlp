# %% [markdown]
# # Module 02: Data Preprocessing Walkthrough
# 
# This notebook demonstrates the preprocessing steps for Romanian text, including cleaning, date normalization, and a comparison of different NLP pipelines.

# %%
# Let's verify our setup
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path.cwd().parent))

# %%
from preprocessing.cleaner import clean_text, parse_romanian_date
from preprocessing.nlp_pipeline import RomanianNLP, get_romanian_stopwords

# %% [markdown]
# ## 1. Text Cleaning and Date Normalization
# We use custom logic for cleaning and `dateparser` for robust Romanian date parsing.

# %%
sample_text = "  În perioada 22 decembrie 2025, comisarii ANPC au aplicat sancțiuni în toată ţara.  "
cleaned = clean_text(sample_text)
date_iso = parse_romanian_date("22 decembrie 2025")

print(f"Original: '{sample_text}'")
print(f"Cleaned:  '{cleaned}'")
print(f"Date ISO: {date_iso}")

# %% [markdown]
# ## 2. Stemming vs Lemmatization
# 
# **Stemming** is a rule-based process that strips suffixes to find the 'root'.
# **Lemmatization** is a dictionary-based process that finds the canonical form (lemma).

# %%
nlp = RomanianNLP()
demo_sentence = "Românii sunt oameni ospitalieri și merg la munte."
nlp.compare_stemming_vs_lemmatization(demo_sentence)

# %% [markdown]
# ## 3. Romanian Stopwords
# Removing frequent words that don't carry much semantic meaning.

# %%
stopwords = get_romanian_stopwords()
print(f"Found {len(stopwords)} Romanian stopwords.")
print(f"Top 20: {stopwords[:20]}")

# %% [markdown]
# ## 4. Comparative NLP Pipelines
# Showing how Stanza and NLTK process the same text.

# %%
print("\nStanza Processing:")
stanza_results = nlp.process_with_stanza("Consumatorii au drepturi.")
for res in stanza_results:
    print(res)

print("\nSpaCy Processing:")
spacy_results = nlp.process_with_spacy("Consumatorii au drepturi.")
for res in spacy_results:
    print(res)

print("\nNLTK Processing (Stemming):")
nltk_results = nlp.process_with_nltk("Consumatorii au drepturi.")
for res in nltk_results:
    print(res)

# %% [markdown]
# ## 5. Dataset Processing
# Finally, we apply our pipeline to the entire dataset.
# The `process_dataset` function orchestrates cleaning, date parsing, and lemmatization for all articles.

# %%
import json
from preprocessing.process_dataset import INPUT_FILE, process_dataset

# Let's look at one raw article before processing
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    print("Raw Article Example:")
    print(json.dumps(raw_data[0], indent=2, ensure_ascii=False)[:500] + "...")

# %%
# Run the full pipeline
process_dataset()

# %%
# Look at the processed result
OUTPUT_FILE = Path.cwd().parent / "data" / "processed" / "articles_anpc_preprocessed.json"
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    processed_data = json.load(f)
    print("\nProcessed Article Example:")
    # Show the new fields added during preprocessing
    example = processed_data[0]
    print(f"Title Cleaned: {example.get('title_cleaned')}")
    print(f"Date ISO: {example.get('date_iso')}")
    print(f"Tokens (first 5): {example.get('content_tokens')[:5]}")
    print(f"Lemmatized Content (snippet): {example.get('lemmatized_content')[:200]}...")
