import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Customer Review Insights", layout="wide")

st.title("🧠 Customer Review Analysis & Insights Generator")
st.write("Explore topics, sentiments, AI-generated summaries, and download automated reports.")

# ----------------------------
# Load Data
# ----------------------------
topics = pd.read_csv("data/topic_labeled.csv")
summaries = pd.read_csv("data/ai_topic_summaries.csv")

# Sentiment data (optional check)
sentiment_path = "data/sentiment_labeled.csv"
if os.path.exists(sentiment_path):
    sentiment_df = pd.read_csv(sentiment_path)
else:
    sentiment_df = None

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.header("Navigation")
choice = st.sidebar.radio(
    "Go to",
    ["Overview", "Topics", "Summaries", "Sentiment", "Report"]
)

# ----------------------------
# Overview Section
# ----------------------------
if choice == "Overview":
    st.subheader("📊 Topic Distribution Overview")
    st.bar_chart(topics["topic"].value_counts())
    st.info(f"Total Reviews: {len(topics):,}")

# ----------------------------
# Topics Section
# ----------------------------
elif choice == "Topics":
    st.subheader("🌀 Topic Wordclouds")
    cols = st.columns(3)
    for i, topic_id in enumerate(sorted(topics["topic"].unique())):
        img_path = f"outputs/topic_wordclouds/topic_{topic_id}_wordcloud.png"
        if os.path.exists(img_path):
            with cols[i % 3]:
                st.image(Image.open(img_path), caption=f"Topic {topic_id}")

# ----------------------------
# Summaries Section
# ----------------------------
elif choice == "Summaries":
    st.subheader("🧩 AI-Generated Topic Summaries")
    for _, row in summaries.iterrows():
        st.markdown(f"### Topic {int(row['topic'])}")
        st.write(row["summary"])
        st.divider()

# ----------------------------
# Sentiment Analysis Section
# ----------------------------
elif choice == "Sentiment":
    st.subheader("💬 Sentiment Analysis Overview")

    if sentiment_df is not None:
        # 1️⃣ Count sentiments
        sentiment_counts = sentiment_df["sentiment"].value_counts()

        # 2️⃣ Bar chart
        st.write("### Sentiment Distribution")
        st.bar_chart(sentiment_counts)

        # 3️⃣ Pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["lightgreen", "lightcoral", "lightgrey"]
        )
        ax.set_title("Sentiment Percentage Split")
        st.pyplot(fig)

        # 4️⃣ Sample reviews section
        st.write("### Sample Reviews by Sentiment")
        sentiment_filter = st.radio("Select Sentiment", ["positive", "neutral", "negative"])
        samples = sentiment_df[sentiment_df["sentiment"] == sentiment_filter].head(10)
        st.dataframe(samples[["review", "sentiment", "compound"]])

    else:
        st.warning("⚠️ Sentiment data not found. Run the sentiment analysis script first.")

# ----------------------------
# Report Download Section
# ----------------------------
elif choice == "Report":
    st.subheader("📄 Download Insights Report")
    pdf_path = "outputs/Customer_Insights_Report.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="Customer_Insights_Report.pdf")
    else:
        st.warning("⚠️ Report not found. Run `generate_report.py` first.")
