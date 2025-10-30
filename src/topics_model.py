import gensim
from gensim import corpora
import pandas as pd
from tqdm import tqdm

def main():
    df = pd.read_csv("data/clean_reviews.csv")

    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]
    df["clean_text"] = df["clean_text"].astype(str)

    texts = [d.split() for d in df["clean_text"]]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    print("Training LDA model...")
    lda = gensim.models.LdaMulticore(
        corpus=corpus, # the document-term matrix
        num_topics=14, # the number of topics to extract
        id2word=dictionary, # mapping of word IDs to words
        passes=5, # number of passes through the corpus during training
        workers=2, # number of worker threads to use
        # chunksize=2000, # number of documents to process in each chunk
        random_state=42 # for reproducibility
    )

    print("\nTop topics discovered:")
    for i, topic in lda.print_topics(num_topics=6):
        print(f"Topic {i}: {topic}")

    print("\nAssigning dominant topic to each document...")
    topics = [
        sorted(lda[c], key=lambda x: -x[1])[0][0] if len(c) > 0 else -1
        for c in tqdm(corpus, desc="Assigning topics")
    ]

    df["topic"] = topics
    df.to_csv("data/topic_labeled.csv", index=False)

    lda.save("models/lda_model.gensim")
    dictionary.save("models/lda_dictionary.gensim")

    print("\nTopic modeling completed.")
    print(df["topic"].value_counts())


if __name__ == "__main__":
    main()
