import streamlit as st
from transformers import pipeline
import spacy

# Load SpaCy model once and cache it
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# Load Hugging Face summarizer with default model (lighter)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")  # uses default small summarizer

# Load sentiment classifier
@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis")  # uses default distilbert sentiment model

# Initialize all models
nlp = load_spacy_model()
summarizer = load_summarizer()
classifier = load_classifier()

# Streamlit UI setup
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
