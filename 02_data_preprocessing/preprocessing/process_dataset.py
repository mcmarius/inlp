import json
from pathlib import Path
from tqdm import tqdm
from preprocessing.cleaner import clean_text, parse_romanian_date
from preprocessing.nlp_pipeline import RomanianNLP

# Setup paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR.parent / "01_data_collection" / "data" / "processed" / "articles_anpc.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "articles_anpc_preprocessed.json"

def process_dataset():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Loading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles.")
    
    # Initialize NLP pipeline (Stanza is used for main processing)
    print("Initialising Stanza NLP pipeline for Romanian...")
    nlp = RomanianNLP()
    
    processed_articles = []
    
    print("Processing articles...")
    for article in tqdm(articles):
        # 1. Clean title and content
        cleaned_title = clean_text(article.get("title", ""))
        cleaned_content = clean_text(article.get("content", ""))
        
        # 2. Parse date to ISO
        date_raw = article.get("date", "")
        date_iso = parse_romanian_date(date_raw)
        
        # 3. NLP enrichment (Lemmatization)
        # We process title and content separately
        title_nlp = nlp.process_with_stanza(cleaned_title)
        content_nlp = nlp.process_with_stanza(cleaned_content)
        
        # Build processed entry
        processed_entry = {
            **article,
            "title_cleaned": cleaned_title,
            "content_cleaned": cleaned_content,
            "date_iso": date_iso,
            "title_tokens": title_nlp,
            "content_tokens": content_nlp,
            "lemmatized_content": " ".join([t["lemma"] for t in content_nlp if t["pos"] != "PUNCT"])
        }
        processed_articles.append(processed_entry)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Success! Processed {len(processed_articles)} articles.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()
