import streamlit as st
from transformers import pipeline
import spacy
import subprocess
import requests
from bs4 import BeautifulSoup
import PyPDF2
from PIL import Image
import pytesseract
import base64
from fpdf import FPDF

# Ensure SpaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Helper: Supported translation models (HuggingFace)
LANG_TO_MODEL = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Russian": "Helsinki-NLP/opus-mt-en-ru",
    "Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "Japanese": "Helsinki-NLP/opus-mt-en-jap",
    "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "Korean": "Helsinki-NLP/opus-mt-en-ko",
    "Turkish": "Helsinki-NLP/opus-mt-en-tr",
    "Bengali": "Helsinki-NLP/opus-mt-en-bn",
    "Greek": "Helsinki-NLP/opus-mt-en-el",
    "Swedish": "Helsinki-NLP/opus-mt-en-sv",
    "Ukrainian": "Helsinki-NLP/opus-mt-en-uk",
    "Polish": "Helsinki-NLP/opus-mt-en-pl",
    "Romanian": "Helsinki-NLP/opus-mt-en-ro",
    "Czech": "Helsinki-NLP/opus-mt-en-cs",
    "Finnish": "Helsinki-NLP/opus-mt-en-fi",
    "Hebrew": "Helsinki-NLP/opus-mt-en-he"
}

# Load models with caching
def get_translator(lang):
    model = LANG_TO_MODEL.get(lang, "Helsinki-NLP/opus-mt-en-fr")
    return pipeline("translation_en_to_{}".format(model.split('-')[-1]), model=model)

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment

summarizer, sentiment = load_models()

# Helper functions
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
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except Exception as e:
        return f"❌ Transcript fetch failed: {str(e)}"

def generate_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Result</a>'

def generate_pdf(summary, sentiment, entities, translation, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Obsidian Protocol Analysis", ln=True, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}")
    pdf.ln(2)
    pdf.multi_cell(0, 10, f"Sentiment: {sentiment['label']} (Confidence: {round(sentiment['score']*100,2)}%)")
    pdf.ln(2)
    ents = ', '.join([f"{e['text']} ({e['label']})" for e in entities])
    pdf.multi_cell(0, 10, f"Entities: {ents if ents else 'None'}")
    pdf.ln(2)
    pdf.multi_cell(0, 10, f"Translation ({lang}):\n{translation}")
    return pdf.output(dest='S').encode('latin1') if isinstance(pdf.output(dest='S'), str) else pdf.output(dest='S')

def inject_confetti():
    st.markdown(
        """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
        if (!window.confettiShown) {
            window.confettiShown = true;
            setTimeout(() => { confetti({ particleCount: 120, spread: 80, origin: { y: 0.6 } }); }, 400);
        }
        </script>
        """,
        unsafe_allow_html=True
    )

# Theme toggle
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

col1, col2 = st.columns([1, 8])
with col1:
    if st.button('🌙' if st.session_state['theme']=='light' else '☀️', key='theme_toggle'):
        st.session_state['theme'] = 'dark' if st.session_state['theme']=='light' else 'light'

# Custom CSS for enhanced glassmorphism and theme
light_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@400;600&display=swap');
body {
    background: radial-gradient(ellipse at 60% 40%, #1a2a24 0%, #0d1a16 100%) !important;
    min-height: 100vh;
    position: relative;
}
body:before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(120deg, rgba(218,165,32,0.08) 0%, rgba(255,255,255,0.01) 100%);
    pointer-events: none;
    z-index: 0;
}
.glass-box {
    margin: 0 auto;
    margin-top: 60px;
    max-width: 800px;
    padding: 3.2rem 2.5rem 2.5rem 2.5rem;
    border-radius: 44px;
    background: rgba(255, 255, 255, 0.18);
    box-shadow: 0 24px 80px 0 rgba(31, 38, 135, 0.22), 0 2px 16px 0 rgba(218,165,32,0.13);
    backdrop-filter: blur(36px);
    -webkit-backdrop-filter: blur(36px);
    border: 2.5px solid rgba(218,165,32,0.32);
    border-top: 2.5px solid #e0eafc;
    border-bottom: 2.5px solid #cfdef3;
    transition: box-shadow 0.3s, border 0.3s;
    animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
    position: relative;
    z-index: 2;
    overflow: hidden;
}
.glass-box:after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 12px;
    background: linear-gradient(90deg, rgba(255,255,255,0.18) 0%, rgba(218,165,32,0.18) 100%);
    opacity: 0.7;
    filter: blur(2px);
    pointer-events: none;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes glassPulse {
    0% { box-shadow: 0 0 0 0 rgba(218,165,32,0.18); }
    70% { box-shadow: 0 0 0 8px rgba(218,165,32,0.08); }
    100% { box-shadow: 0 0 0 0 rgba(218,165,32,0.18); }
}
@keyframes ripple {
    0% { transform: scale(1); }
    50% { transform: scale(0.96); }
    100% { transform: scale(1); }
}
@keyframes cardSpringIn {
    0% { opacity: 0; transform: translateY(40px) scale(0.98); }
    60% { opacity: 1; transform: translateY(-8px) scale(1.03); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes shimmerMove {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes iconWiggle {
    0% { transform: rotate(0deg) scale(1); }
    20% { transform: rotate(-8deg) scale(1.08); }
    40% { transform: rotate(8deg) scale(1.08); }
    60% { transform: rotate(-4deg) scale(1.04); }
    80% { transform: rotate(4deg) scale(1.04); }
    100% { transform: rotate(0deg) scale(1); }
}
.glass-search textarea:focus {
    animation: glassPulse 0.7s;
}
.analyze-btn:active {
    animation: ripple 0.3s;
}
.glass-card {
    animation: cardSpringIn 0.9s cubic-bezier(.39,.575,.56,1.000);
    will-change: transform, box-shadow;
}
.glass-card:hover, .analyze-btn:hover {
    box-shadow: 0 8px 40px rgba(218,165,32,0.18), 0 2px 16px 0 rgba(255,255,255,0.18);
    transform: translateY(-4px) scale(1.025);
}
.glass-shimmer {
    position: absolute;
    top: 0; left: 0; right: 0; height: 100%;
    background: linear-gradient(120deg, rgba(255,255,255,0.12) 0%, rgba(218,165,32,0.08) 100%);
    opacity: 0.5;
    pointer-events: none;
    z-index: 10;
    animation: shimmerMove 4s linear infinite;
}
.icon-laurel:hover, .icon-feather:hover, .icon-key:hover, .icon-lock:hover {
    animation: iconWiggle 0.7s;
}
.glass-title {
    text-align: center;
    font-size: 3.1rem;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    color: #e6d7b0;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
    text-shadow: 0 2px 24px rgba(218,165,32,0.18), 0 1px 0 #fff;
}
.glass-sub {
    text-align: center;
    font-size: 1.28rem;
    color: #b6c6b0;
    margin-bottom: 2.1rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    opacity: 0.92;
}
.glass-search {
    background: rgba(255,255,255,0.32);
    border-radius: 28px;
    box-shadow: 0 4px 32px rgba(31,38,135,0.10), 0 1.5px 8px 0 rgba(218,165,32,0.13);
    padding: 1.4rem 1.4rem 0.7rem 1.4rem;
    margin-bottom: 2.2rem;
    border: 1.5px solid rgba(218,165,32,0.22);
    position: relative;
    z-index: 3;
    overflow: hidden;
}
.glass-search:before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 8px;
    background: linear-gradient(90deg, rgba(255,255,255,0.18) 0%, rgba(218,165,32,0.18) 100%);
    opacity: 0.7;
    filter: blur(1.5px);
    pointer-events: none;
}
.glass-search textarea {
    border-radius: 22px !important;
    background: rgba(255,255,255,0.96) !important;
    border: 2px solid #e0eafc !important;
    font-size: 1.22rem !important;
    padding: 1.3rem !important;
    min-height: 120px !important;
    font-family: 'Montserrat', sans-serif;
    box-shadow: 0 2px 12px rgba(31,38,135,0.06);
    transition: box-shadow 0.2s;
}
.analyze-btn {
    width: 100%;
    border-radius: 22px;
    background: linear-gradient(90deg, #bfa14a 0%, #e6d7b0 100%);
    color: #222;
    font-weight: 700;
    font-size: 1.22rem;
    padding: 1rem 0;
    margin-top: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(218,165,32,0.13);
    border: none;
    transition: background 0.2s, box-shadow 0.2s;
    cursor: pointer;
    letter-spacing: 0.5px;
    font-family: 'Montserrat', sans-serif;
}
.result-section {
    margin-top: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    justify-content: center;
    animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
}
.glass-card {
    background: rgba(255,255,255,0.38);
    border-radius: 28px;
    box-shadow: 0 4px 32px rgba(218,165,32,0.10), 0 1.5px 8px 0 rgba(255,255,255,0.13);
    padding: 1.3rem 1.3rem 1rem 1.3rem;
    min-width: 260px;
    max-width: 340px;
    flex: 1 1 320px;
    border: 2px solid rgba(218,165,32,0.22);
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    z-index: 4;
    overflow: hidden;
}
.glass-card:before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 8px;
    background: linear-gradient(90deg, rgba(255,255,255,0.18) 0%, rgba(218,165,32,0.18) 100%);
    opacity: 0.7;
    filter: blur(1.5px);
    pointer-events: none;
}
.result-title {
    font-size: 1.22rem;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    color: #bfa14a;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5em;
    text-shadow: 0 1px 8px rgba(218,165,32,0.13);
}
.result-content {
    font-size: 1.08rem;
    color: #222;
    font-family: 'Montserrat', sans-serif;
    margin-bottom: 0.5rem;
    background: none;
    box-shadow: none;
    border-radius: 0;
    padding: 0;
}
.icon-laurel { color: #bfa14a; font-size: 1.2em; margin-right: 0.2em; }
.icon-feather { color: #bfa14a; font-size: 1.2em; margin-right: 0.2em; }
.icon-key { color: #bfa14a; font-size: 1.2em; margin-right: 0.2em; }
.icon-lock { color: #bfa14a; font-size: 1.2em; margin-right: 0.2em; }
</style>
"""
dark_css = """
<style>
body {
    background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
}
.glass-box {
    margin: 0 auto;
    margin-top: 60px;
    max-width: 700px;
    padding: 2.8rem 2.2rem 2.2rem 2.2rem;
    border-radius: 36px;
    background: rgba(30, 34, 42, 0.38);
    box-shadow: 0 16px 64px 0 rgba(31, 38, 135, 0.22);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 2px solid rgba(255, 255, 255, 0.12);
    transition: box-shadow 0.3s;
    animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: translateY(0); }
}
.glass-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 900;
    color: #e0eafc;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
    font-family: 'Segoe UI', 'Montserrat', sans-serif;
    text-shadow: 0 2px 12px rgba(31,38,135,0.18);
}
.glass-sub {
    text-align: center;
    font-size: 1.22rem;
    color: #b6c6e6;
    margin-bottom: 2.1rem;
    font-family: 'Segoe UI', 'Montserrat', sans-serif;
    opacity: 0.85;
}
.glass-search textarea {
    border-radius: 20px !important;
    background: rgba(30,34,42,0.92) !important;
    border: 2px solid #3a3a4a !important;
    font-size: 1.22rem !important;
    padding: 1.3rem !important;
    min-height: 120px !important;
    font-family: 'Segoe UI', 'Montserrat', sans-serif;
    color: #e0eafc !important;
    box-shadow: 0 2px 12px rgba(31,38,135,0.10);
    transition: box-shadow 0.2s;
}
.glass-search textarea:focus {
    box-shadow: 0 4px 24px rgba(31,38,135,0.23);
    border: 2px solid #b6c6e6 !important;
}
.analyze-btn {
    width: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #232526 0%, #414345 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1.22rem;
    padding: 1rem 0;
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
    background: linear-gradient(90deg, #414345 0%, #232526 100%);
    box-shadow: 0 4px 24px rgba(102,126,234,0.18);
}
.result-section {
    margin-top: 1.5rem;
    animation: fadeIn 1.2s cubic-bezier(.39,.575,.56,1.000);
}
.result-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #e0eafc;
    margin-bottom: 0.5rem;
    font-family: 'Segoe UI', 'Montserrat', sans-serif;
    display: flex;
    align-items: center;
    gap: 0.5em;
}
.result-content {
    background: rgba(30,34,42,0.75);
    border-radius: 16px;
    padding: 1.2rem 1.1rem;
    margin-bottom: 1.1rem;
    font-size: 1.12rem;
    color: #e0eafc;
    font-family: 'Segoe UI', 'Montserrat', sans-serif;
    box-shadow: 0 1px 6px rgba(31,38,135,0.14);
    transition: background 0.2s;
}
</style>
"""

st.markdown(light_css if st.session_state['theme']=='light' else dark_css, unsafe_allow_html=True)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.markdown('<div class="glass-title">🧠 Obsidian Protocol</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-sub">Reveal, Decode, Translate. AI-powered media insights.</div>', unsafe_allow_html=True)

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

# Unified glassmorphism search box with all input types and results
with st.form("analyze_form"):
    input_mode = st.radio("📥 Choose input type:", [
        "📝 Text", "📎 Upload File", "🌐 Article URL", "▶️ YouTube Video", "🖼️ Image (OCR)"
    ], horizontal=True)
    input_text = ""
    if input_mode == "📝 Text":
        input_text = st.text_area("", key="glass_search", help="Enter text, article, or speech here.", placeholder="Paste any text to analyze...", height=140)
    elif input_mode == "📎 Upload File":
        uploaded_file = st.file_uploader("📎 Upload a .txt or .pdf", type=["txt", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                input_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
    elif input_mode == "🌐 Article URL":
        url = st.text_input("🌐 Enter Article URL")
        if url:
            input_text = extract_text_from_url(url)
    elif input_mode == "▶️ YouTube Video":
        yt_url = st.text_input("▶️ Enter YouTube Link")
        if yt_url:
            input_text = extract_transcript_from_youtube(yt_url)
    elif input_mode == "🖼️ Image (OCR)":
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
            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            sentiment_result = sentiment(input_text)[0]
            doc = nlp(input_text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            translator = get_translator(lang)
            translated = translator(summary)[0]["translation_text"]
        inject_confetti()
        inject_copy_js()
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">✨ Summary <button onclick="copyToClipboard(\'' + summary.replace("'", "\\'") + '\')" style="margin-left:8px;cursor:pointer;">📋</button></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{summary}</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">📊 Sentiment <button onclick="copyToClipboard(\'' + sentiment_result["label"] + '\')" style="margin-left:8px;cursor:pointer;">📋</button></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{sentiment_result["label"]} <br> <span style="font-size:0.98rem;opacity:0.7;">Confidence: {round(sentiment_result["score"]*100, 2)}%</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">🧠 Named Entities <button onclick="copyToClipboard(\'' + ', '.join([e['text'] for e in entities]).replace("'", "\\'") + '\')" style="margin-left:8px;cursor:pointer;">📋</button></div>', unsafe_allow_html=True)
        if entities:
            ents_html = ''.join([f'<span style="display:inline-block;background:rgba(102,126,234,0.13);border-radius:8px;padding:0.3em 0.7em;margin:0.18em 0.3em 0.18em 0;font-size:1.01rem;font-weight:500;color:#4b3fa7;">{e["text"]} <span style="font-size:0.92rem;opacity:0.7;">({e["label"]})</span></span>' for e in entities])
            st.markdown(f'<div class="result-content">{ents_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-content">No named entities found.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">🌍 {lang} Translation <button onclick="copyToClipboard(\'' + translated.replace("'", "\\'") + '\')" style="margin-left:8px;cursor:pointer;">📋</button></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-content">{translated}</div>', unsafe_allow_html=True)
        # PDF Download
        pdf_bytes = generate_pdf(summary, sentiment_result, entities, translated, lang)
        st.download_button("📄 Download PDF", data=pdf_bytes, file_name="obsidian_result.pdf", mime="application/pdf")
        # TXT Download
        result = f"Summary:\n{summary}\n\nSentiment: {sentiment_result['label']} ({round(sentiment_result['score'], 2)})\n\nEntities: {entities}"
        st.markdown(generate_download_link(result, "obsidian_result.txt"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown('</div>', unsafe_allow_html=True)
