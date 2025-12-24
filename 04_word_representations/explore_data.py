import pandas as pd
import os
from collections import Counter

script_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(script_dir, "../02_data_preprocessing/data/processed/articles_anpc_preprocessed.json"))
df = pd.read_json(data_path)

sectors = {
    "Energie": df[df['content'].str.contains("energie", case=False)],
    "Alimentar": df[df['content'].str.contains("alimentar", case=False)],
    "Bancar": df[df['content'].str.contains("bancar|credit", case=False, regex=True)]
}

for name, d in sectors.items():
    print(f"\nSector: {name} ({len(d)} articles)")
    all_text = " ".join(d['content'].tolist()).lower()
    words = [w for w in all_text.split() if len(w) > 5]
    common = Counter(words).most_common(20)
    print(f"Common words: {common}")

for target in ["control", "preț", "contract"]:
    print(f"\nChecking for '{target}'...")
    for name, d in sectors.items():
        count = d['content'].str.contains(target, case=False).sum()
        print(f"Articles in {name} containing '{target}': {count}")
