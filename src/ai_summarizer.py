import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os

# === Setup ===
os.makedirs("data", exist_ok=True)

print("Loading summarization model (pszemraj/long-t5-tglobal-base-16384-book-summary)...")
model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
summarizer = pipeline("summarization", model=model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Load Data ===
df = pd.read_csv("data/topic_labeled.csv")
grouped = df.groupby("topic").head(500)

# === Helper: Chunk text safely ===
def chunk_text_by_tokens(text, max_tokens=15000):
    """Split text into sub-strings that fit model context window."""
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        chunk = tokenizer.decode(sub_tokens, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

# === Summarization Loop ===
summaries = []

for topic, group in tqdm(grouped.groupby("topic"), desc="Summarizing each topic"):
    full_text = " ".join(group["clean_text"].astype(str).tolist())
    
    # If text is empty, skip
    if not full_text.strip():
        print(f"Skipping topic {topic}: empty text.")
        continue

    # Create token-safe chunks
    chunks = chunk_text_by_tokens(full_text)
    topic_summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_length=150,
                min_length=60,
                do_sample=False
            )[0]["summary_text"]
            topic_summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk for topic {topic}: {e}")

    # Combine chunk summaries
    if topic_summaries:
        combined_summary = " ".join(topic_summaries)
        summaries.append({"topic": topic, "summary": combined_summary})
    else:
        print(f"No summaries generated for topic {topic}")

# === Save Results ===
summary_df = pd.DataFrame(summaries)
summary_df.to_csv("data/ai_topic_summaries.csv", index=False)

print("\nTopic summaries saved to: data/ai_topic_summaries.csv")
