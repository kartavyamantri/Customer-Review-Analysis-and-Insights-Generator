# 🧠 Customer Review Analysis & Insights Generator

> “Your customers are already telling you what they want - all you have to do is listen.”  

Welcome to the **Customer Review Analysis & Insights Generator** - an end-to-end NLP and Generative AI project that transforms thousands of messy customer reviews into structured insights, visualizations, and AI-generated summaries.  

This project bridges the gap between **data analytics** and **Generative AI**, helping businesses decode customer opinions and make smarter, data-driven decisions.

---

## 🚀 Project Overview

The goal of this project is to take **raw textual customer reviews** (e.g., from Amazon, Flipkart, etc.) and:
- Clean and preprocess them using NLP techniques  
- Identify hidden themes and discussion patterns via **Topic Modeling (LDA)**  
- Measure **topic coherence** to ensure interpretability  
- Perform **Sentiment Analysis** to understand customer emotions  
- Generate **AI-based summaries** for each topic using **Transformer models**  
- Visualize insights interactively via a **Streamlit Dashboard**

In short, it’s like turning an unstructured mess of text into a beautiful, data-backed story your product team can act on.

---

## 🧩 Key Features

✅ **Text Preprocessing & Cleaning**  
Removes noise, punctuation, and stopwords; lemmatizes words for consistency.  

🧠 **Topic Modeling (LDA)**  
Extracts dominant themes from reviews such as “Product Quality,” “Delivery Experience,” or “Customer Support.”  

📊 **Topic Coherence Evaluation**  
Automatically finds the optimal number of topics for best interpretability.  

💬 **Sentiment Analysis**  
Quantifies how customers feel about each topic (positive/negative/neutral).  

🤖 **AI-Powered Summarization**  
Generates concise summaries per topic using **Hugging Face Transformer models**.  

📈 **Interactive Streamlit Dashboard**  
Visualizes insights - word clouds, topic distributions, and sentiment trends - all in one place.  

---

## 🛠️ Tech Stack

**Languages & Frameworks:**  
- Python  
- Streamlit  

**Libraries Used:**  
- **Data Processing:** `pandas`, `numpy`, `re`, `os`  
- **NLP & Topic Modeling:** `nltk`, `spacy`, `gensim`  
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`, `pyLDAvis`  
- **AI Summarization:** `transformers`, `huggingface_hub`, `tqdm`  
- **Dashboard:** `streamlit`  

---

## 🧠 Learning Objectives

This project helped build a strong foundation in:
- Data preprocessing and NLP pipelines  
- Topic modeling and text-based unsupervised learning  
- Applying GenAI summarization models  
- Creating interactive dashboards for analytics storytelling  

It serves as a **milestone project** in the journey from data analysis to **Generative AI application development**.

---

## ⚙️ Installation & Setup

### Step 1️⃣: Clone the Repository
```bash
git clone https://github.com/kartavyamantri/Customer-Review-Analysis-and-Insights-Generator.git
cd Customer-Review-Analysis-and-Insights-Generator

### Step 2️⃣: Create a Virtual Environment
python -m venv venv
venv/Scripts/Activate.ps1  # For Windows

### Step 3️⃣: Install Required Dependencies
pip install -r requirements.txt
```

---

## 🧮 Usage Instructions

```bash
### Step 1: Clean and Preprocess Data
python src/data_preprocessing.py

### Step 2: Generate Topic with LDA
python src/topic_modeling.py

### Step 3: Find Optimal Topics using Coherence
python src/coherence_optimization.py

### Step 4: Visualize Wordclouds & Sentiments
python src/topic_visualization.py
python src/sentiment_analysis.py

### Step 5: Generate AI Summaries
python src/ai_summarizer.py

### Step 6: Launch Streamlit Dashboard
streamlit run app/app.py
```

---

## 🧠 Model Details

### 🧩 Topic Modeling (LDA)

Library: gensim

Parameters: num_topics = 14 (optimal via coherence score)

Output: topic keywords, word distribution, and dominant topic per review

### 🤖 Summarization Model

Model: pszemraj/long-t5-tglobal-base-16384-book-summary

Framework: Hugging Face Transformers

Task: Generates concise summaries for each topic by processing large volumes of customer reviews

Token Handling: Supports up to 16,384 tokens, enabling summarization of long text chunks without excessive splitting

## 📈 Dashboard Preview

The Streamlit dashboard includes:

- Sidebar for topic selection
- Topic-wise wordclouds
- Sentiment distribution graphs
- AI-generated topic summaries
- Review exploration and filters

