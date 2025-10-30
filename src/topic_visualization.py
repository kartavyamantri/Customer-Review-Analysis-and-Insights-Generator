import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import os
import gensim
from gensim import corpora
import pickle

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load Labeled Data
# -----------------------------
df = pd.read_csv("data/topic_labeled.csv")

# -----------------------------
# Topic Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(x="topic", data=df, hue="topic", palette="pastel", legend=False)
plt.title("Topic Distribution in Amazon Reviews")
plt.xlabel("Topic ID")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/topic_distribution.png")
plt.close()

print("Saved: outputs/topic_distribution.png")

# -----------------------------
# Generate WordClouds for Each Topic
# -----------------------------
print("\nGenerating topic wordclouds...")

texts = [d.split() for d in df["clean_text"].astype(str)]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(t) for t in texts]

lda = gensim.models.LdaModel.load("models/lda_model.gensim") if os.path.exists("models/lda_model.gensim") else None

if lda:
    for topic_id in tqdm(range(lda.num_topics), desc="Creating wordclouds"):
        topic_words = dict(lda.show_topic(topic_id, topn=40))
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(topic_words)
        wc.to_file(f"outputs/topic_wordclouds/topic_{topic_id}_wordcloud.png")

    print("Wordclouds saved in outputs/topic_wordclouds/ folder.")
else:
    print("LDA model not found in models/. Wordclouds skipped.")
