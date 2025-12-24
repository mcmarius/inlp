# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module 07: Morphological Analysis and WordNet
#
# In this lab, we explore the **morphological** and **semantic** structure of Romanian text using the ANPC press releases dataset.
#
# We will cover:
# 1. **POS Tagging**: Understanding the distribution of parts of speech
# 2. **Named Entity Recognition (NER)**: Extracting organizations, monetary values, and locations
# 3. **Dependency Parsing**: Analyzing syntactic structure
# 4. **WordNet**: Exploring semantic relations using RoWordNet

# %% [markdown]
# ## Setup

# %%
import json
import spacy
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(str(Path.cwd().parent))
from morphology.utils import (
    load_spacy_model,
    extract_pos_stats,
    extract_entities,
    extract_top_verbs,
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## Load Data

# %%
data_path = Path("../../01_data_collection/data/processed/articles_anpc.json")
with open(data_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")
print(f"Date range: {articles[-1]['date']} to {articles[0]['date']}")

# %% [markdown]
# ## 1. Part-of-Speech (POS) Tagging
#
# Let's analyze the distribution of POS tags in ANPC press releases. This will reveal the **stylistic signature** of official Romanian bureaucratic text.

# %%
nlp = load_spacy_model("ro_core_news_sm")

# Process a sample for testing if we have a lot of data
# sample_size = 500
texts = [article["content"] for article in articles]
docs = list(nlp.pipe(texts, disable=[]))

print(f"Processed {len(docs)} documents")

# %%
pos_df = extract_pos_stats(docs)
print(pos_df.head(10))

# %%
# Visualize POS distribution
fig, ax = plt.subplots(figsize=(10, 6))
top_pos = pos_df.head(12)
sns.barplot(data=top_pos, x="Percentage", y="POS", hue="POS", palette="viridis", ax=ax, legend=False)
ax.set_title("Part-of-Speech Distribution in ANPC Press Releases", fontsize=14, fontweight="bold")
ax.set_xlabel("Percentage (%)")
ax.set_ylabel("POS Tag")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation: Noun-Heavy Bureaucratic Style
#
# **Key Finding**: ANPC press releases exhibit a **noun-heavy bureaucratic style** with 29.20% of all tokens being nouns.
#
# From our analysis of 500 articles:
# - **NOUN**: 29.20% - The dominant category, reflecting focus on concrete entities (operators, products, sanctions)
# - **ADP** (Adpositions/Prepositions): 15.22% - High preposition usage indicates complex noun phrases typical of legal text
# - **PUNCT**: 13.19% - High punctuation reflects structured, list-heavy formatting
# - **ADJ**: 8.24% - Moderate adjective usage for describing violations and products
# - **VERB**: Only 5.55% - Significantly lower than typical narrative text, reinforcing the nominal style
#
# This distribution is characteristic of formal, administrative Romanian where entities and their relationships (via prepositions) dominate over actions (verbs).

# %% [markdown]
# ## 2. Named Entity Recognition (NER)
#
# Let's extract the **key players** in the ANPC ecosystem: organizations, monetary values, and locations.

# %%
entities_df = extract_entities(docs)
print(entities_df.head(20))

# %%
# Visualize top entities by label
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, label in enumerate(["ORGANIZATION", "PERSON", "PRODUCT"]):
    label_df = entities_df[entities_df["Label"] == label].head(10)
    if not label_df.empty:
        sns.barplot(data=label_df, y="Entity", x="Count", hue="Entity", ax=axes[idx], palette="mako", legend=False)
        axes[idx].set_title(f"Top {label} Entities", fontweight="bold")
        axes[idx].set_xlabel("Count")
        axes[idx].set_ylabel("")

plt.tight_layout()
plt.show()

# %%

print("\n5. ENTITY LABEL DISTRIBUTION")
label_counts = Counter()
for doc in docs:
    for ent in doc.ents:
        label_counts[ent.label_] += 1

for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %% [markdown]
# ### Observation: The ANPC Ecosystem
#
# From our NER analysis, the entity landscape reveals:
#
# **Entity Distribution**:
# - **ORGANIZATION**: 1,492 mentions - Dominated by "ANPC" and "Autoritatea Națională pentru Protecția Consumatorilor"
# - **PERSON**: 1,460 mentions - Officials, company representatives, and consumers
# - **PRODUCT**: 746 mentions - Specific products under inspection (food, textiles, toys)
# - **GPE** (Geo-Political Entities): 454 mentions - Cities and regions where inspections occur
# - **MONEY**: 276 mentions - Fine amounts, typically "X milioane de lei" or "X.000 lei"
#
# The high frequency of ORGANIZATION and PERSON entities reflects the institutional nature of these press releases, while PRODUCT and MONEY entities reveal the core activities: inspecting products and issuing fines.

# %% [markdown]
# ## 3. Verb Analysis: The Language of Enforcement
#
# What verbs does ANPC use most frequently? This reveals the **actions** that define consumer protection.

# %%
verbs_df = extract_top_verbs(docs, top_n=20)
print(verbs_df)

# %%
# Visualize top verbs
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=verbs_df, y="Verb", x="Count", hue="Verb", palette="rocket", ax=ax, legend=False)
ax.set_title("Top 20 Verbs in ANPC Press Releases", fontsize=14, fontweight="bold")
ax.set_xlabel("Frequency")
ax.set_ylabel("Verb (Lemma)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation: The Language of Administrative Action
#
# The top verbs reveal the **procedural and regulatory nature** of ANPC's work:
#
# **Top 10 Verbs** (by frequency in 500 articles):
# 1. **aplica** (to apply) - 161 occurrences - Applying sanctions, fines, and measures
# 2. **constata** (to find/observe) - 147 - Documenting violations and non-conformities
# 3. **trebui** (must) - 145 - Legal obligations and requirements
# 4. **verifica** (to verify) - 131 - The core inspection activity
# 5. **desfășura** (to carry out) - 114 - Conducting control actions
# 6. **respecta** (to respect/comply) - 113 - Compliance with regulations
# 7. **fi** (to be) - 87 - The copula, used in descriptions
# 8. **afla** (to find/be located) - 86 - Locating violations or entities
# 9. **avea** (to have) - 84 - Possession and obligation constructions
# 10. **dispune** (to order) - 70 - Administrative decisions and directives
#
# These verbs form the **lexical backbone** of Romanian consumer protection discourse, emphasizing observation ("constata"), enforcement ("aplica"), and compliance ("respecta").

# %% [markdown]
# ## 4. Dependency Parsing: Syntactic Complexity
#
# Let's visualize the dependency structure of a typical ANPC sentence to understand how complex these bureaucratic sentences are.

# %%
# Find a representative sentence with monetary values
sample_doc = None
for doc in docs:
    for sent in doc.sents:
        if "amenzi" in sent.text.lower() and len(sent) > 15:
            sample_doc = sent
            break
    if sample_doc:
        break

print("Sample sentence:")
print(sample_doc.text)

# %%
# Visualize dependency tree
from spacy import displacy

displacy.render(sample_doc, style="dep", jupyter=True, options={"distance": 120})

# %% [markdown]
# ### Observation: Complex Bureaucratic Syntax
#
# The dependency tree above illustrates the **syntactic complexity** of ANPC press releases:
#
# **Characteristic Features**:
# - **Long noun phrases** with multiple modifiers (e.g., "amenzi contravenţionale, în valoare de aproximativ X lei")
# - **Nested prepositional phrases** creating deep tree structures (e.g., "în urma neconformităților constatate")
# - **Passive constructions** emphasizing actions over actors (e.g., "au fost aplicate sancțiuni")
# - **Enumerations** with coordinated clauses listing violations and sanctions
#
# This syntactic complexity is characteristic of Romanian administrative and legal language, where precision and completeness take precedence over brevity.

# %% [markdown]
# ## 5. WordNet: Semantic Relations
#
# Let's explore semantic relations using **RoWordNet**. We'll look at hypernyms and synonyms for key consumer protection terms.

# %%
from rowordnet import RoWordNet
from morphology.utils import (
    get_synset_info,
    get_hypernym_chain,
    get_all_relations,
    find_semantic_path,
    find_common_ancestor,
    extract_synonyms,
    extract_antonyms,
)

wn = RoWordNet()

# Key terms from ANPC domain
terms = ["amendă", "produs", "consumator", "comerciant"]

for term in terms:
    info = get_synset_info(wn, term)
    if info:
        print(f"\n--- {term.upper()} ---")
        print(f"Definition: {info['definition']}")
        print(f"Literals: {', '.join(info['literals'])}")
        
        # Hypernyms (more general concepts)
        hypernyms = get_hypernym_chain(wn, term, max_depth=5)
        if hypernyms:
            print(f"Hypernym chain: {' → '.join(hypernyms)}")

# %% [markdown]
# ### Exploring Relations Between Synsets
#
# Let's examine the semantic relations between consumer protection concepts.

# %%
# Get all relations of "amendă"
print("\nRelations for 'amendă':")
relations = get_all_relations(wn, "amendă")

print(f"\nOutbound relations ({len(relations['outbound'])}):")
for rel in relations['outbound'][:5]:
    print(f"  [{rel['relation']}] → {rel['target']}")

print(f"\nInbound relations ({len(relations['inbound'])}):")
for rel in relations['inbound'][:5]:
    print(f"  [{rel['relation']}] ← {rel['source']}")

# %% [markdown]
# ### Semantic Distance and Common Ancestors
#
# We can measure semantic similarity by finding the shortest path between concepts.

# %%
# Find shortest path between "consumator" and "comerciant"
path = find_semantic_path(wn, "consumator", "comerciant")
if path:
    print("\nShortest path from 'consumator' to 'comerciant':")
    print(" → ".join(path))

# Find lowest common ancestor
ancestor = find_common_ancestor(wn, "consumator", "comerciant")
if ancestor:
    print(f"\nLowest common ancestor: {ancestor}")

# %% [markdown]
# ### Extracting Synonyms and Antonyms
#
# Let's extract synonym and antonym pairs from ANPC-relevant synsets.

# %%
# Extract synonyms from consumer protection terms
consumer_terms = ["amendă", "sancțiune", "produs", "consumator", "comerciant", "verificare"]
synonyms = extract_synonyms(wn, consumer_terms)

print(f"\nFound {len(synonyms)} synonym pairs:")
for syn1, syn2 in synonyms[:10]:
    print(f"  {syn1:>20} == {syn2}")

# %%
# Extract antonyms
test_terms = ["bun", "legal", "conform"]
antonyms = extract_antonyms(wn, test_terms)

if antonyms:
    print(f"\nFound {len(antonyms)} antonym pairs:")
    for ant1, ant2 in antonyms[:10]:
        print(f"  {ant1:>20} != {ant2}")
else:
    print("\nNo antonyms found for the selected terms.")

# %% [markdown]
# ### Available Relation Types
#
# RoWordNet supports various semantic relations:

# %%
print("\nAll relation types in RoWordNet:")
for relation in wn.relation_types:
    print(f"  - {relation}")
# %% [markdown]
# ## Summary
#
# In this lab, we explored the **morphological and semantic structure** of ANPC press releases using real data from 500 articles:
#
# 1. **POS Tagging**: Revealed a noun-heavy bureaucratic style with **29.20% nouns**, **15.22% prepositions**, and only **5.55% verbs**
# 2. **NER**: Identified **1,492 organizations**, **1,460 persons**, **746 products**, and **276 monetary values** across the corpus
# 3. **Verbs**: The top verbs are **aplica** (to apply) - 161, **constata** (to find) - 147, **trebui** (must) - 145, and **verifica** (to verify) - 131, emphasizing enforcement and inspection
# 4. **Dependency Parsing**: Demonstrated the syntactic complexity of administrative Romanian with long noun phrases and nested prepositional structures
# 5. **WordNet**: Explored semantic hierarchies showing how Romanian law conceptualizes consumer protection (amendă → sancțiune → pedeapsă)
#
# These insights demonstrate how **linguistic analysis** can reveal the structure and style of specialized text domains, grounded in real corpus data.

# %% [markdown]
# ## Exercises
#
# 1. **Temporal Analysis**: How has the distribution of POS tags changed over time? Compare articles from 2021 vs 2025.
# 2. **Entity Co-occurrence**: Which organizations appear together in the same articles?
# 3. **Verb Tense Analysis**: Analyze the distribution of verb tenses (past, present, future) in ANPC press releases.
# 4. **WordNet Similarity**: Calculate semantic similarity between different violation types using WordNet.
