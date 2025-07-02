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

# Custom CSS for glassmorphism search box
st.markdown(
    """
    <style>
    .glass-box {
        margin: 0 auto;
        margin-top: 60px;
        max-width: 600px;
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .glass-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 0.5rem;
    }
    .glass-sub {
        text-align: center;
        font-size: 1.1rem;
        color: #444;
        margin-bottom: 1.5rem;
    }
    .glass-search textarea {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.7) !important;
        border: 1px solid #ddd !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        min-height: 120px !important;
    }
    .stButton>button {
        border-radius: 16px;
        background: #222;
        color: #fff;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.7rem 2.2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.markdown('<div class="glass-title">🧠 Obsidian Protocol</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-sub">Reveal the truth behind news, speeches, or social media posts.</div>', unsafe_allow_html=True)

# Glassmorphism search box
with st.form("analyze_form"):
    text = st.text_area("Paste any text to analyze:", key="glass_search", help="Enter text, article, or speech here.")
    submitted = st.form_submit_button("🔍 Analyze")

if 'submitted' not in locals():
    submitted = False

if submitted:
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

st.markdown('</div>', unsafe_allow_html=True)
