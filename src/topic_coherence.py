import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_coherence_values(dictionary, corpus, texts, start=4, limit=20, step=2):
    """
    Compute c_v coherence for various number of topics.
    """
    coherence_values = []
    model_list = []
    
    for num_topics in tqdm(range(start, limit + 1, step), desc="Evaluating topics"):
        model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=5,
            workers=2,
            random_state=42
        )
        coherencemodel = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherencemodel.get_coherence()
        coherence_values.append(coherence_score)
        model_list.append(model)
    
    return model_list, coherence_values


def main():
    print("📊 Loading dataset...")
    df = pd.read_csv("data/clean_reviews.csv")
    df = df.sample(n=500000, random_state=42)
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]
    df["clean_text"] = df["clean_text"].astype(str)
    
    texts = [d.split() for d in df["clean_text"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    print("\n🔍 Computing coherence scores for topic range...")
    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        texts=texts,
        start=4,
        limit=20,
        step=2
    )

    # Plot coherence values
    x = list(range(4, 21, 2))
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o')
    plt.title('LDA Topic Coherence Optimization')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score (c_v)')
    plt.grid(True)
    plt.savefig("outputs/coherence_scores.png")
    plt.show()

    # Print results
    for m, cv in zip(x, coherence_values):
        print(f"Num Topics = {m}, Coherence Score = {round(cv, 4)}")

    # Best number of topics
    best_idx = coherence_values.index(max(coherence_values))
    best_num_topics = x[best_idx]
    print(f"\n✅ Optimal number of topics: {best_num_topics}")

if __name__ == "__main__":
    main()
