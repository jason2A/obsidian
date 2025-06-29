import streamlit as st
from transformers import pipeline
import spacy
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load spaCy model
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
        summary = summarizer(user_input, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
        sentiment = classifier(user_input)[0]
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

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

