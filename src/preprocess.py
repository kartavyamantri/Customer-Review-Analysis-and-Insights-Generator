import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tqdm import tqdm

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text): # Basic text cleaning for Amazon reviews
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

def load_txt_dataset(path): # Load .txt file; auto-detect if labels exist.
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    data = []
    for line in tqdm(lines, desc="Reading lines"):
        parts = re.split(r'\s', line, maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in ["__label__2", "__label__1"]:
            label, review = parts
            data.append({"label": label.strip(), "review": review.strip()})
        else:
            data.append({"label": None, "review": line.strip()})
    return pd.DataFrame(data)

if __name__ == "__main__":
    path = "data/amazon_reviews.txt"
    df = load_txt_dataset(path)
    print(f"Loaded {len(df)} reviews.")

    tqdm.pandas()
    df["clean_text"] = df["review"].progress_apply(clean_text)

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/train.txt", sep="\t", index=False)
    test_df.to_csv("data/test.txt", sep="\t", index=False)
    df.to_csv("data/clean_reviews.csv", index=False)

    print("✅ Data cleaned and split. Saved in data/ folder.")
