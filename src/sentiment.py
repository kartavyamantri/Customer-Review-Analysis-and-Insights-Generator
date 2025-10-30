import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm

nltk.download('vader_lexicon')
tqdm.pandas()

sid = SentimentIntensityAnalyzer()
df = pd.read_csv("data/clean_reviews.csv")

df["compound"] = df["clean_text"].progress_apply(lambda x: sid.polarity_scores(str(x))["compound"])
df["sentiment"] = df["compound"].apply(lambda c: "positive" if c > 0.1 else ("negative" if c < -0.1 else "neutral"))

df.to_csv("data/sentiment_labeled.csv", index=False)

print("✅ Sentiment analysis completed.")
print(df["sentiment"].value_counts())
