# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 06: Transformers Applications
#
# In Lab 4, we introduced Word Embeddings (Static & Contextual).
# In this lab, we dive into **Transformers** and their applications beyond just embeddings.
#
# We will cover:
# 1.  **Theory**: From RNNs to Attention.
# 2.  **Summarization**: 
#     - **0-shot Extractive**: Using DistilBERT (Unsupervised).
#     - **Fine-tuned Abstractive**: Using T5 (Supervised).
# 3.  **Semantic Search**: Building a Sentence Transformer.
# 4.  **Topic Modeling**: Advanced clustering with BERTopic.
#
# Grounded in the **ANPC Dataset**, we will try to find patterns in consumer warnings and generate titles for articles.

# %%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import sys
import os

# NLP Libraries
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from sentence_transformers import SentenceTransformer, models
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import evaluate

# Add parent to path for utils
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent.parent))

from notebook_utils import path_resolver
from utils import get_improved_stopwords

# Aesthetic setup
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Theoretical Motivation: RNNs vs Transformers
#
# ### Recurrent Neural Networks (RNNs) & LSTMs
# Before Transformers (2017), **Recurrent Neural Networks (RNNs)** were the state-of-the-art for NLP.
#
# *   **Mechanism**: Process tokens sequentially ($t_1, t_2, ... t_n$). The hidden state $h_t$ depends on $h_{t-1}$.
# *   **Problem 1**: **Sequentiality**. You cannot parallelize training. $t_{100}$ must wait for $t_{99}$.
# *   **Problem 2**: **Long-term Dependencies**. Information from $t_1$ often vanishes by the time we reach $t_{100}$, even with LSTMs (Long Short-Term Memory).
#
# ### The Attention Mechanism
# "Attention is All You Need" (Vaswani et al., 2017) changed everything.
# *   **Idea**: Instead of processing sequentially, let every token **"attend"** (look at) every other token at once.
# *   **Result**: Paralllelizable and capable of capturing global context instantly.
#
# We will see this power in action with **Summarization** and **Semantic Matching**.

# %% [markdown]
# ## 2. Data Loading
#
# We load our ANPC articles. We will use the 'title' and 'content' fields.

# %%
DATA_FILE = Path("../02_data_preprocessing/data/processed/articles_anpc_preprocessed.json")

df = pd.read_json(path_resolver(DATA_FILE, external=True))
# Filter out empty titles or content
df = df.dropna(subset=['title', 'content'])
df = df[df['content'].str.len() > 100] # Keep substantial articles

print(f"Loaded {len(df)} articles.")
print(df[['title', 'content']].head(3))

# %% [markdown]
# ## 3. Application 1: Text Summarization
#
# **Task**: Given the content of a press release, generate a suitable title.
#
# ### 3.1 Method A: 0-shot Extractive Summary (TextRank with DistilBERT)
#
# Since we want to use our lightweight **DistilBERT** from Lab 4 (`racai/distilbert-base-romanian-cased`), we faces a challenge: it's an **Encoder-only** model. It cannot generate text 0-shot (like GPT).
#
# However, we can use it to build a powerful **Extractive Summarizer**.
# 1.  Split text into sentences.
# 2.  Compute the embedding for each sentence (using DistilBERT).
# 3.  Compute the Similarity Matrix (how similar is every sentence to every other sentence?).
# 4.  Apply **PageRank**. Sentences that are similar to many other sentences are "central" to the topic.
# 5.  Select the top ranked sentences.

# %%
MODEL_NAME_BERT = "racai/distilbert-base-romanian-cased"
tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
model_bert = AutoModel.from_pretrained(MODEL_NAME_BERT).to(device)

def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling (average of all token embeddings)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def extractive_summary(text, top_n=1):
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 10]
    
    if len(sentences) <= top_n:
        return text
        
    embeddings = [get_sentence_embedding(s, model_bert, tokenizer_bert) for s in sentences]
    
    # Similarity Matrix
    sim_mat = cosine_similarity(embeddings)
    
    # Graph: Nodes = Sentences, Edges = Similarity
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Return top N
    return " ".join([s for _, s in ranked_sentences[:top_n]])

# Test on a sample
sample_idx = 100
sample_text = df.iloc[sample_idx]['content']
sample_title = df.iloc[sample_idx]['title']

print(f"Original Title: {sample_title}")
print(f"---")
print(f"Extractive Summary (Top 1 sentence):\n{extractive_summary(sample_text, top_n=1)}")

# %% [markdown]
# ### 3.2 Method B: Fine-tuned Abstractive Summary (T5)
#
# To generate *new* text (Abstractive), we need a **Sequence-to-Sequence (Seq2Seq)** model.
# We will use `dumitrescustefan/t5-v1_1-base-romanian`, a T5 model pre-trained on Romanian text.
#
# **Note on Resources**: Fine-tuning T5 requires significant GPU memory. We will run a "tiny" training loop (few samples, few steps) just to demonstrate the code pipeline. In a real scenario, you would train on the full corpus for epochs.

# %%
# Prepare Dataset: Input = Content, Target = Title
train_df, val_df = train_test_split(df[['content', 'title']], test_size=0.1, random_state=42)

# Use a tiny subset for lab demonstration speed
train_df_tiny = train_df.iloc[:50] # Only 50 examples
val_df_tiny = val_df.iloc[:10]

MODEL_NAME_T5 = "dumitrescustefan/t5-v1_1-base-romanian"
tokenizer_t5 = AutoTokenizer.from_pretrained(MODEL_NAME_T5)
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_T5)
model_t5.resize_token_embeddings(len(tokenizer_t5))
model_t5.to(device)

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["content"]]
    model_inputs = tokenizer_t5(inputs, max_length=256, truncation=True)

    labels = tokenizer_t5(text_target=examples["title"], max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Use HuggingFace Dataset object for trainer
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df_tiny)
val_dataset = Dataset.from_pandas(val_df_tiny)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer_t5, model=model_t5)

# Metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # batch_decode expects integers, and numpy 2.0+ can be strict or weird with types. 
    # Explicitly casting to lists of integers avoids specialized numpy types causing overflows in C extensions.
    
    # 1. Handle Predictions
    # Ensure it's a list or numpy array
    if isinstance(predictions, list):
        predictions_arr = np.array(predictions)
    else:
        predictions_arr = predictions

    # Safety: replace any -100 with pad token just in case
    predictions_arr = np.where(predictions_arr != -100, predictions_arr, tokenizer_t5.pad_token_id)
    
    # Convert to standard Python int list to avoid any numpy type issues in batch_decode
    decoded_preds = tokenizer_t5.batch_decode(predictions_arr.tolist(), skip_special_tokens=True)
    
    # 2. Handle Labels
    if isinstance(labels, list):
        labels_arr = np.array(labels)
    else:
        labels_arr = labels
        
    labels_arr = np.where(labels_arr != -100, labels_arr, tokenizer_t5.pad_token_id)
    decoded_labels = tokenizer_t5.batch_decode(labels_arr.tolist(), skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    prediction_lens = [np.count_nonzero(pred != tokenizer_t5.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_t5_summ",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2, # Small batch for CPU/Colab
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3, # Small epoch count for demo
    predict_with_generate=True,
    logging_steps=5,
    use_cpu=not torch.cuda.is_available()
)

trainer = Seq2SeqTrainer(
    model=model_t5,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer_t5,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting Fine-tuning (Demo Mode)...")
trainer.train()

# %% [markdown]
# ### 3.3 Evaluation
# Let's generate a title for an unseen article.

# %%
idx = 0
sample_text_val = val_df_tiny.iloc[idx]['content']
true_title = val_df_tiny.iloc[idx]['title']

input_ids = tokenizer_t5("summarize: " + sample_text_val, return_tensors="pt", max_length=256, truncation=True).input_ids.to(device)
outputs = model_t5.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
gen_title = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

print(f"Original Text (Snippet): {sample_text_val[:200]}...")
print(f"True Title: {true_title}")
print(f"Generated Title: {gen_title}")

# %% [markdown]
# ### 3.4 Understanding T5 Results
#
# **Note on Title Generation Quality**: The generated title above is likely nonsensical or repetitive. This is expected given our training constraints:
#
# *   **Tiny Training Set**: Only 50 examples (vs. thousands needed for good generalization)
# *   **Limited Epochs**: 3 epochs (vs. 10-20+ for production models)
# *   **Small Model**: Base model size (vs. large variants)
# *   **Domain Specificity**: ANPC press releases have a very specific style that requires more examples to learn
#
# **For Production Use**: You would need to:
# 1.  Train on the full dataset (200+ articles)
# 2.  Use more epochs (10-20) with early stopping
# 3.  Consider a larger T5 variant if compute allows
# 4.  Potentially augment with synthetic data or transfer learning from news summarization
#
# The code pipeline demonstrated here is correct and production-ready; only the scale needs adjustment.

# %% [markdown]
# ## 4. Application 2: Sentence Transformers (Semantic Search)
#
# "Sentence Transformers" (SBERT) modify the BERT architecture to create semantically meaningful sentence embeddings that can be compared using cosine similarity.
#
# We will construct a Sentence Transformer using our trusty `racai/distilbert-base-romanian-cased`.
# **Task**: Find potentially duplicate articles or recurring warnings (e.g., "Atenție la produsele lactate").

# %%
# Define the Sentence Transformer
# 1. Transformer model
word_embedding_model = models.Transformer(MODEL_NAME_BERT)
# 2. Pooling (mean of all token vectors)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# 3. Assemble
sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=str(device))

print("Encoding ANPC corpus for semantic search (this may take a moment)...")
# Let's take a subset of ~200 top articles to save time
df_subset = df.head(500).copy()
embeddings_sbert = sbert_model.encode(df_subset['content'].tolist(), show_progress_bar=True)

# %%
# Find duplicates within the subset
# We look for pairs with cosine similarity > 0.95
print("Searching for duplicates...")
duplicates = []
sim_matrix = cosine_similarity(embeddings_sbert)
np.fill_diagonal(sim_matrix, 0) # Ignore self-match

# Iterate simply (upper triangle)
for i in range(len(sim_matrix)):
    for j in range(i + 1, len(sim_matrix)):
        if sim_matrix[i, j] > 0.95:
            duplicates.append((i, j, sim_matrix[i, j]))

duplicates = sorted(duplicates, key=lambda x: x[2], reverse=True)

print(f"Found {len(duplicates)} pairs with >95% similarity.")
for i, j, score in duplicates[:5]:
    print(f"\n[Score: {score:.4f}]")
    print(f"1: {df_subset.iloc[i]['title']}")
    print(f"2: {df_subset.iloc[j]['title']}")

# %% [markdown]
# ## 5. Application 3: BERTopic
#
# BERTopic is a topic modeling technique that leverages transformers and class-based TF-IDF (`c-TF-IDF`).
#
# **Advantage**: It produces much more coherent topics than LDA because it understands semantic context.
# **Task**: Analyze the "General" category articles to see what distinct topics exist inside.

# %%
# Setup BERTopic
# We pass our pre-calculated sentence transformer, but BERTopic wraps it nicely usually.
# However, explicit embedding is often stable.

# Filter for a specific category to make it interesting, or just use the subset
# Let's use the subset we already encoded
docs = df_subset['content'].tolist()

# Configure CountVectorizer to filter stopwords for more meaningful topics
# Without this, topics would be dominated by common words like "de", "în", "și", "la"
stopwords = get_improved_stopwords()
vectorizer_model = CountVectorizer(
    stop_words=stopwords,
    min_df=2,  # Ignore terms that appear in fewer than 2 documents
    ngram_range=(1, 2)  # Include unigrams and bigrams for richer topics
)

topic_model = BERTopic(
    embedding_model=sbert_model,
    vectorizer_model=vectorizer_model,  # Apply stopword filtering
    min_topic_size=3, # Small dataset -> small topics
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings_sbert)

# Visualize Topics
topic_info = topic_model.get_topic_info()
print(topic_info.head(10))

# Interactive Visualization (will render in Notebook)
# topic_model.visualize_topics() # Commented out to avoid rendering in non-interactive run, but key for real user

# %% [markdown]
# ### 5.1 Temporal BERTopic (Topics over Time)
# If we have timestamps, we can see how topics evolve.

# %%
# Check for possible date columns
timestamps = df_subset['date_iso'].tolist()

topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=10)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=5)
# print("Temporal topics calculated.")

# %% [markdown]
# ![](topics_over_time.png)

# %% [markdown]
# ## 6. Insights & Conclusions
#
# *   **Summarization**:
#     *   **Extractive (DistilBERT)**: Successfully identifies central sentences using PageRank. Often selects key quotes or statements from officials that capture the article's main message.
#     *   **Abstractive (T5)**: With our tiny training set (50 examples, 3 epochs), the model produces nonsensical output. As documented in section 3.4, production use requires the full dataset and significantly more training.
# *   **Semantic Search**: Found 10,330+ pairs with >95% similarity in our subset. Many are exact duplicates (score=1.0) due to repeated press releases, plus near-duplicates from similar control campaigns across different time periods.
# *   **BERTopic**: After applying stopword filtering, revealed meaningful topics including:
#     *   **Food Products** (Topic 0: "unor_produse_au_alimentare") - food safety inspections
#     *   **Airline Services** (Topic 5: "air_wizz_wizz air") - Wizz Air consumer complaints
#     *   **Energy Services** (Topic 6: "privind_consumatorilor_energie") - energy consumer protection
#     *   **Accessibility** (Topic 8: "dizabilități_persoane_transport") - disability rights and transport
#
# The stopword filtering was crucial - without it, topics were dominated by "de", "în", "și", "la" and provided no meaningful insights.
#
# ## Exercises
#
# 1.  **Title Generation**: Train the T5 model on the full dataset with more epochs. Compare ROUGE scores between training on all categories vs. category-specific models.
# 2.  **Semantic Search**: Build a "Related Articles" recommender. Given an article ID, return the top 5 most semantically similar articles (excluding exact duplicates).
# 3.  **Cross-Lingual**: If you used a multilingual model (like `xlm-roberta`), could you find English articles similar to these Romanian ones? (Research exercise).

