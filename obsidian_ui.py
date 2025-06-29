import streamlit as st
import subprocess
import sys

# ✅ Ensure the SpaCy model is downloaded BEFORE importing spacy
def ensure_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

ensure_spacy_model()

# ✅ Now import spacy safely
import spacy
from transformers import pipeline

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load HuggingFace pipelines
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit UI
st.set_page_config(page_title="Obsidian Protocol", layout="wide")

st.title("🧠 Obsidian Protocol")
st.subheader("Reveal the truth behind any media, speech, or post.")

user_input = st.text_area("Paste your article, speech, or social post here:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some text to analyze.")
    else:
        # Generate summary
        summary = summarizer(user_input, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

        # Get sentiment
        sentiment = classifier(user_input)[0]

        # Named Entity Recognition
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Display results
        st.markdown("### 🧠 Summary")
        st.info(summary)

        st.markdown("### 🎭 Sentiment")
        st.success(f"**Label:** {sentiment['label']}, **Confidence:** {round(sentiment['score'], 2)}")

        st.markdown("### 🕵️ Key Entities")
        if entities:
            for entity, label in entities:
                st.write(f"• **{entity}** ({label})")
        else:
            st.write("No named entities found.")

        
