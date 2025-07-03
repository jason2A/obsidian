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
import subprocess

# 🧠 CONFIG
st.set_page_config(page_title="🧠 Obsidian Protocol v2.0", layout="wide")
st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0e1117;
            color: white;
        }
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

# LOAD MODELS
@st.cache_resource
def load_models():
    # Ensure SpaCy model is downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    return nlp, summarizer, classifier, translator

nlp, summarizer, classifier, translator = load_models()

# HELPERS
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

# 🧩 UI
st.title("🧠 Obsidian Protocol v2.0")
st.caption("👓 Reveal, Decode, Translate. AI-powered media insights.")
st.markdown("---")

# Custom CSS for advanced glassmorphism and beautiful UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;
    }
    .glass-box {
        margin: 0 auto;
        margin-top: 60px;
        max-width: 650px;
        padding: 2.8rem 2.2rem 2.2rem 2.2rem;
        border-radius: 32px;
        background: rgba(255, 255, 255, 0.22);
        box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.18);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1.5px solid rgba(255, 255, 255, 0.28);
        transition: box-shadow 0.3s;
        animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .glass-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a1a2e;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
        text-shadow: 0 2px 12px rgba(31,38,135,0.08);
    }
    .glass-sub {
        text-align: center;
        font-size: 1.18rem;
        color: #3a3a4a;
        margin-bottom: 2.1rem;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
        opacity: 0.85;
    }
    .glass-search textarea {
        border-radius: 18px !important;
        background: rgba(255,255,255,0.82) !important;
        border: 1.5px solid #e0eafc !important;
        font-size: 1.18rem !important;
        padding: 1.2rem !important;
        min-height: 120px !important;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
        box-shadow: 0 2px 12px rgba(31,38,135,0.06);
        transition: box-shadow 0.2s;
    }
    .glass-search textarea:focus {
        box-shadow: 0 4px 24px rgba(31,38,135,0.13);
        border: 1.5px solid #b6c6e6 !important;
    }
    .analyze-btn {
        width: 100%;
        border-radius: 18px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        font-weight: 700;
        font-size: 1.18rem;
        padding: 0.9rem 0;
        margin-top: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(102,126,234,0.13);
        border: none;
        transition: background 0.2s, box-shadow 0.2s;
        cursor: pointer;
        letter-spacing: 0.5px;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
    }
    .analyze-btn:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 24px rgba(102,126,234,0.18);
    }
    .result-section {
        margin-top: 1.5rem;
        animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
    }
    .result-content {
        background: rgba(255,255,255,0.65);
        border-radius: 14px;
        padding: 1.1rem 1rem;
        margin-bottom: 1.1rem;
        font-size: 1.08rem;
        color: #333;
        font-family: 'Segoe UI', 'Montserrat', sans-serif;
        box-shadow: 0 1px 6px rgba(31,38,135,0.04);
        transition: background 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.markdown('<div class="glass-title">🧠 Obsidian Protocol</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-sub">Reveal the truth behind news, speeches, or social media posts. Paste your text and experience instant insight.</div>', unsafe_allow_html=True)

# Add animated placeholder text using JavaScript injection
animated_placeholder_js = """
<script>
const prompts = [
  'Paste any text to analyze...',
  'Try a news article, speech, or tweet!',
  'What do you want to summarize today?',
  'Drop in your thoughts or a story...'
];
let i = 0;
let j = 0;
let currentPrompt = '';
let isDeleting = false;
const textarea = document.querySelector('textarea[data-testid="stTextArea"]');
function typePrompt() {
  if (!textarea) return;
  if (!isDeleting && j <= prompts[i].length) {
    currentPrompt = prompts[i].substring(0, j++);
    textarea.setAttribute('placeholder', currentPrompt);
    setTimeout(typePrompt, 60);
  } else if (isDeleting && j >= 0) {
    currentPrompt = prompts[i].substring(0, j--);
    textarea.setAttribute('placeholder', currentPrompt);
    setTimeout(typePrompt, 30);
  } else {
    isDeleting = !isDeleting;
    if (!isDeleting) {
      i = (i + 1) % prompts.length;
    }
    setTimeout(typePrompt, isDeleting ? 800 : 1200);
  }
}
setTimeout(typePrompt, 800);
</script>
"""
st.markdown(animated_placeholder_js, unsafe_allow_html=True)

# Glassmorphism search box with integrated features
with st.form("analyze_form"):
    input_mode = st.radio("📥 Choose input type:", ["Text", "Upload File", "Article URL", "YouTube Video", "Image (OCR)"])
    input_text = ""
    if input_mode == "Text":
        input_text = st.text_area("", key="glass_search", help="Enter text, article, or speech here.", placeholder="Paste any text to analyze...", height=140)
    elif input_mode == "Upload File":
        uploaded_file = st.file_uploader("📎 Upload a .txt or .pdf", type=["txt", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                input_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
    elif input_mode == "Article URL":
        url = st.text_input("🌐 Enter Article URL")
        if url:
            input_text = extract_text_from_url(url)
    elif input_mode == "YouTube Video":
        yt_url = st.text_input("▶️ Enter YouTube Link")
        if yt_url:
            input_text = extract_transcript_from_youtube(yt_url)
    elif input_mode == "Image (OCR)":
        uploaded_img = st.file_uploader("🖼️ Upload Image with Text", type=["jpg", "png"])
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            input_text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text", value=input_text, height=150)
    submitted = st.form_submit_button("✨ Analyze", use_container_width=True)

if 'submitted' not in locals():
    submitted = False

if submitted:
    if input_text.strip():
        with st.spinner("Analyzing with Obsidian Engine..."):
            # Summarize
            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            # Sentiment
            sentiment_result = classifier(input_text)[0]
            # NER
            doc = nlp(input_text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            # Translation
            translated = translator(summary)[0]["translation_text"]

        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">✨ Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{summary}</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-title">📊 Sentiment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{sentiment_result["label"]} <br> <span style="font-size:0.98rem;opacity:0.7;">Confidence: {round(sentiment_result["score"]*100, 2)}%</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="result-title">🧠 Named Entities</div>', unsafe_allow_html=True)
        if entities:
            ents_html = ''.join([f'<span style="display:inline-block;background:rgba(102,126,234,0.13);border-radius:8px;padding:0.3em 0.7em;margin:0.18em 0.3em 0.18em 0;font-size:1.01rem;font-weight:500;color:#4b3fa7;">{e["text"]} <span style="font-size:0.92rem;opacity:0.7;">({e["label"]})</span></span>' for e in entities])
            st.markdown(f'<div class="result-content">{ents_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-content">No named entities found.</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-title">🌍 French Translation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{translated}</div>', unsafe_allow_html=True)

        result = f"Summary:\n{summary}\n\nSentiment: {sentiment_result['label']} ({round(sentiment_result['score'], 2)})\n\nEntities: {entities}"
        st.markdown(generate_download_link(result, "obsidian_result.txt"), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("⚡ Obsidian Protocol — Advanced Insight Engine inspired by NotebookLM, GPT-4o, and AI tools.")
