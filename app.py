import streamlit as st
from transformers import pipeline
import spacy
import subprocess

# Ensure SpaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load models with caching
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment

summarizer, sentiment = load_models()

# App layout
st.set_page_config(page_title="🧠 Obsidian Protocol", layout="centered")
st.title("🧠 Obsidian Protocol")
st.subheader("Reveal the truth behind news, speeches, or social media posts.")

text = st.text_area("Paste any text to analyze:")

if st.button("🔍 Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            # Summarize
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            # Sentiment
            sentiment_result = sentiment(text)[0]
            # NER
            doc = nlp(text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Output
        st.success("Analysis Complete!")
        st.write("### ✨ Summary")
        st.info(summary)

        st.write("### 📊 Sentiment")
        st.json(sentiment_result)

        st.write("### 🧠 Named Entities")
        st.json(entities)
    else:
        st.warning("Please enter some text to analyze.")
