import streamlit as st
from transformers import pipeline
import spacy
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
import PyPDF2
from PIL import Image
import pytesseract
import base64

# 🧠 UI CONFIG
st.set_page_config(page_title="Obsidian Protocol", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .main { font-family: 'Helvetica Neue', sans-serif; background-color: #0e1117; color: white; }
        .stButton>button {
            background-color: #222;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }
        .stTextArea textarea {
            background-color: #1e222a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ LOAD MODELS
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    return nlp, summarizer, classifier, translator

nlp, summarizer, classifier, translator = load_models()

# 🧾 UTILITIES
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_url(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))
    except:
        return "❌ Unable to extract text."

def extract_transcript_from_youtube(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except Exception as e:
        return f"❌ Transcript fetch failed: {str(e)}"

def generate_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Result</a>'

# 🎨 APP UI
st.title("🧠 Obsidian Protocol")
st.subheader("Reveal the truth behind any media, speech, or post — powered by transformers and vision AI.")

input_mode = st.radio("Choose input type:", ["Text", "Upload File", "Article URL", "YouTube Video", "Image (OCR)"])
input_text = ""

# 📥 GET TEXT BASED ON INPUT TYPE
if input_mode == "Text":
    input_text = st.text_area("Paste your article, speech, or social post:")

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            input_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            input_text = uploaded_file.read().decode("utf-8")

elif input_mode == "Article URL":
    url = st.text_input("Enter article URL:")
    if url:
        input_text = extract_text_from_url(url)

elif input_mode == "YouTube Video":
    yt_url = st.text_input("Enter YouTube link:")
    if yt_url:
        input_text = extract_transcript_from_youtube(yt_url)

elif input_mode == "Image (OCR)":
    uploaded_img = st.file_uploader("Upload image with text", type=["jpg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_text = pytesseract.image_to_string(image)
        st.text_area("Extracted Text", value=input_text, height=150)

# 🔍 ANALYZE
if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please provide valid input.")
    else:
        with st.spinner("Analyzing with Obsidian Engine..."):
            # 🔎 NLP Tasks
            summary = summarizer(input_text, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]
            sentiment = classifier(input_text[:512])[0]
            doc = nlp(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            translated = translator(summary)[0]["translation_text"]

        # 📊 OUTPUT
        st.markdown("### 🧠 Summary")
        st.info(summary)

        st.markdown("### 🌍 French Translation")
        st.info(translated)

        st.markdown("### 🎭 Sentiment")
        st.success(f"**Label:** {sentiment['label']} | **Confidence:** {round(sentiment['score'], 2)}")

        st.markdown("### 🕵️ Named Entities")
        if entities:
            for entity, label in entities:
                st.markdown(f"• **{entity}** — `{label}`")
        else:
            st.info("No named entities found.")

        final_result = f"Summary:\n{summary}\n\nSentiment: {sentiment['label']} ({round(sentiment['score'],2)})\n\nEntities: {entities}"
        st.markdown(generate_download_link(final_result, "obsidian_result.txt"), unsafe_allow_html=True)

# ℹ️ FOOTER
st.markdown("---")
st.caption("⚡ Obsidian Protocol — AI-powered media analyzer built with Streamlit, Transformers, SpaCy, and Vision AI.")
