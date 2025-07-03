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
import re

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
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return "❌ youtube_transcript_api is not installed. Please install it to use this feature."
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
    return pdf.output(dest='S')

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

# Niagara Launcher style CSS and sidebar
niagara_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
body {
    background: linear-gradient(120deg, #232526 0%, #414345 100%) !important;
    min-height: 100vh;
    font-family: 'Montserrat', sans-serif;
}
.niagara-sidebar {
    position: fixed;
    top: 0; left: 0; bottom: 0;
    width: 72px;
    background: rgba(0,0,0,0.18);
    box-shadow: 2px 0 24px 0 rgba(0,0,0,0.10);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 40px;
    z-index: 100;
}
.niagara-sidebar .icon-btn {
    width: 54px; height: 54px;
    margin-bottom: 32px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    background: rgba(0,0,0,0.10);
    box-shadow: 0 2px 12px rgba(0,0,0,0.10);
    border: none;
    transition: box-shadow 0.18s, background 0.18s;
    cursor: pointer;
    position: relative;
}
.niagara-sidebar .icon-btn:hover, .niagara-sidebar .icon-btn.active {
    background: rgba(0,0,0,0.22);
    box-shadow: 0 4px 24px 0 rgba(0,255,128,0.13);
}
.niagara-sidebar svg {
    width: 32px; height: 32px;
    stroke: #111;
    stroke-width: 2.4;
    fill: none;
    transition: stroke 0.18s;
}
.niagara-sidebar .icon-btn.active svg, .niagara-sidebar .icon-btn:hover svg {
    stroke: #00c896;
}
@media (prefers-color-scheme: dark) {
    .niagara-sidebar svg { stroke: #fff; }
    .niagara-sidebar .icon-btn.active svg, .niagara-sidebar .icon-btn:hover svg { stroke: #00c896; }
}
.glass-box {
    margin: 0 auto;
    margin-top: 80px;
    max-width: 700px;
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 36px;
    background: rgba(0,0,0,0.18);
    box-shadow: 0 16px 64px 0 rgba(0,0,0,0.22);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: none;
    transition: box-shadow 0.3s;
    position: relative;
    z-index: 2;
    overflow: hidden;
}
.glass-title {
    text-align: center;
    font-size: 2.5rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    color: #fff;
    letter-spacing: 1.2px;
    margin-bottom: 1.5rem;
    text-shadow: 0 2px 16px rgba(0,0,0,0.10);
}
.glass-search {
    background: rgba(0,0,0,0.22);
    border-radius: 32px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.10);
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    margin-bottom: 2.2rem;
    border: none;
    position: relative;
    z-index: 3;
    display: flex;
    align-items: center;
    justify-content: center;
}
.glass-search textarea {
    border-radius: 24px !important;
    background: rgba(0,0,0,0.13) !important;
    border: none !important;
    font-size: 1.3rem !important;
    padding: 1.3rem !important;
    min-height: 90px !important;
    font-family: 'Montserrat', sans-serif;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.glass-search textarea:focus {
    box-shadow: 0 4px 24px rgba(0,255,128,0.13);
}
.analyze-btn {
    width: 100%;
    border-radius: 24px;
    background: linear-gradient(90deg, #00c896 0%, #232526 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1.22rem;
    padding: 1rem 0;
    margin-top: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,255,128,0.13);
    border: none;
    transition: background 0.2s, box-shadow 0.2s;
    cursor: pointer;
    letter-spacing: 0.5px;
    font-family: 'Montserrat', sans-serif;
}
.analyze-btn:hover {
    background: linear-gradient(90deg, #232526 0%, #00c896 100%);
    box-shadow: 0 4px 24px rgba(0,255,128,0.18);
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
    background: rgba(0,0,0,0.22);
    border-radius: 28px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.10);
    padding: 1.3rem 1.3rem 1rem 1.3rem;
    min-width: 260px;
    max-width: 340px;
    flex: 1 1 320px;
    border: none;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    z-index: 4;
    overflow: hidden;
    color: #fff;
}
.glass-card:hover {
    box-shadow: 0 8px 40px rgba(0,255,128,0.18), 0 2px 16px 0 rgba(0,0,0,0.18);
    transform: translateY(-4px) scale(1.025);
}
.result-title {
    font-size: 1.18rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    color: #00c896;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5em;
    text-shadow: 0 1px 8px rgba(0,255,128,0.13);
}
.result-content {
    font-size: 1.08rem;
    color: #fff;
    font-family: 'Montserrat', sans-serif;
    margin-bottom: 0.5rem;
    background: none;
    box-shadow: none;
    border-radius: 0;
    padding: 0;
}
</style>
<div class="niagara-sidebar">
  <div class="icon-btn active" title="Search">
    <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
  </div>
  <div class="icon-btn" title="Upload">
    <svg viewBox="0 0 24 24"><rect x="4" y="17" width="16" height="3" rx="1.5"/><polyline points="12 17 12 3"/><polyline points="7 8 12 3 17 8"/></svg>
  </div>
  <div class="icon-btn" title="Article/Link">
    <svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="13" rx="2.5"/><line x1="8" y1="3" x2="16" y2="3"/><line x1="12" y1="3" x2="12" y2="7"/></svg>
  </div>
  <div class="icon-btn" title="YouTube">
    <svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="4"/><polygon points="10 9 16 12 10 15 10 9"/></svg>
  </div>
  <div class="icon-btn" title="Image/OCR">
    <svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="8" cy="10" r="2.2"/><polyline points="21 19 15 13 9 19 3 13"/></svg>
  </div>
  <div class="icon-btn" title="Settings">
    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.09a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </div>
</div>
"""
st.markdown(niagara_css, unsafe_allow_html=True)

# Add a glassy vertical sidebar with Nothing OS–style icons
sidebar_css = """
<style>
.nothing-sidebar {
    position: fixed;
    top: 0; left: 0; bottom: 0;
    width: 68px;
    background: rgba(255,255,255,0.13);
    box-shadow: 2px 0 24px 0 rgba(31,38,135,0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1.5px solid rgba(218,165,32,0.13);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 32px;
    z-index: 100;
}
.nothing-sidebar .icon-btn {
    width: 44px; height: 44px;
    margin-bottom: 18px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 16px;
    background: rgba(255,255,255,0.10);
    border: 1.5px solid rgba(218,165,32,0.10);
    transition: box-shadow 0.18s, border 0.18s, background 0.18s;
    cursor: pointer;
    position: relative;
}
.nothing-sidebar .icon-btn:hover, .nothing-sidebar .icon-btn.active {
    background: rgba(255,255,255,0.22);
    border: 1.5px solid #bfa14a;
    box-shadow: 0 2px 16px 0 rgba(218,165,32,0.13);
}
.nothing-sidebar svg {
    width: 26px; height: 26px;
    stroke: #bfa14a;
    stroke-width: 2.2;
    fill: none;
    transition: stroke 0.18s;
}
.nothing-sidebar .icon-btn.active svg {
    stroke: #1a2a24;
}
</style>
"""

# Add Streamlit state for sidebar mode
if 'sidebar_mode' not in st.session_state:
    st.session_state['sidebar_mode'] = 'search'

# Sidebar icon mapping
sidebar_modes = [
    ('search', 'Search', '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>'),
    ('upload', 'Upload', '<svg viewBox="0 0 24 24"><rect x="4" y="17" width="16" height="3" rx="1.5"/><polyline points="12 17 12 3"/><polyline points="7 8 12 3 17 8"/></svg>'),
    ('link', 'Article/Link', '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="13" rx="2.5"/><line x1="8" y1="3" x2="16" y2="3"/><line x1="12" y1="3" x2="12" y2="7"/></svg>'),
    ('youtube', 'YouTube', '<svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="4"/><polygon points="10 9 16 12 10 15 10 9"/></svg>'),
    ('image', 'Image/OCR', '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="8" cy="10" r="2.2"/><polyline points="21 19 15 13 9 19 3 13"/></svg>'),
    ('settings', 'Settings', '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.09a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>')
]

# Sidebar HTML with JS for interactivity
sidebar_html = '<div class="nothing-sidebar">'
for mode, label, icon_svg in sidebar_modes:
    active = 'active' if st.session_state['sidebar_mode'] == mode else ''
    sidebar_html += f'<div class="icon-btn {active}" title="{label}" onclick="window.parent.postMessage(\'sidebar_mode:{mode}\', \'*\')">{icon_svg}</div>'
sidebar_html += '</div>'
sidebar_js = '''<script>
window.addEventListener('message', function(e) {
  if (typeof e.data === 'string' && e.data.startsWith('sidebar_mode:')) {
    const mode = e.data.split(':')[1];
    window.location.hash = 'sidebar_mode_' + mode;
    window.dispatchEvent(new Event('hashchange'));
  }
});
window.addEventListener('hashchange', function() {
  const mode = window.location.hash.replace('#sidebar_mode_', '');
  if (mode) {
    window.parent.postMessage('set_streamlit_sidebar_mode:' + mode, '*');
  }
});
</script>'''
st.markdown(sidebar_css + sidebar_html + sidebar_js, unsafe_allow_html=True)

# Streamlit: update sidebar_mode from hash (simulate JS->Python)
import streamlit.  components.v1 as components
sidebar_mode = st.session_state['sidebar_mode']
if '_sidebar_mode' not in st.session_state:
    st.session_state['_sidebar_mode'] = sidebar_mode
components.html('''<script>
window.addEventListener('message', function(e) {
  if (typeof e.data === 'string' && e.data.startsWith('set_streamlit_sidebar_mode:')) {
    const mode = e.data.split(':')[1];
    window.parent.document.dispatchEvent(new CustomEvent('streamlit:set_sidebar_mode', {detail: mode}));
  }
});
window.parent.document.addEventListener('streamlit:set_sidebar_mode', function(e) {
  window.parent.postMessage('streamlit_set_sidebar_mode:' + e.detail, '*');
});
</script>''', height=0)

# Main UI: only sidebar and glass search/results
st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.markdown('<div class="glass-shimmer"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-title" style="margin-bottom:2.5rem;"><span class="icon-laurel">&#127807;</span> Obsidian Search <span class="icon-laurel">&#127807;</span></div>', unsafe_allow_html=True)

with st.form("analyze_form"):
    st.markdown('<div class="glass-search" style="display:flex;align-items:center;justify-content:center;position:relative;">'
                '<span style="font-size:1.7rem;margin-right:0.7em;opacity:0.7;">🔍</span>', unsafe_allow_html=True)
    input_text = ""
    if st.session_state['sidebar_mode'] == 'search':
        input_text = st.text_area("", key="glass_search", help="Type, paste, or drop anything to analyze...", placeholder="Search or analyze anything...", height=80, label_visibility="collapsed")
    elif st.session_state['sidebar_mode'] == 'upload':
        uploaded_file = st.file_uploader("Upload a .txt or .pdf", type=["txt", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                input_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
    elif st.session_state['sidebar_mode'] == 'link':
        url = st.text_input("Enter Article URL")
        if url:
            input_text = extract_text_from_url(url)
    elif st.session_state['sidebar_mode'] == 'youtube':
        yt_url = st.text_input("Enter YouTube Link")
        if yt_url:
            input_text = extract_transcript_from_youtube(yt_url)
    elif st.session_state['sidebar_mode'] == 'image':
        uploaded_img = st.file_uploader("Upload Image with Text", type=["jpg", "png"])
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            input_text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text", value=input_text, height=150)
    st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("✨ Analyze", use_container_width=True)

if 'submitted' not in locals():
    submitted = False

# 2. Auto-detect input type
def detect_input_type(text):
    if re.match(r'^https?://(www\.)?youtube\.com/watch\?v=', text):
        return 'youtube'
    elif re.match(r'^https?://', text):
        return 'url'
    elif text.strip().endswith('.pdf') or text.strip().endswith('.txt'):
        return 'file'
    else:
        return 'text'

# 3. Show loader/progress bar on submit
if submitted:
    if input_text.strip():
        with st.spinner("Analyzing with Obsidian Engine..."):
            input_type = detect_input_type(input_text.strip())
            if input_type == 'youtube':
                processed_text = extract_transcript_from_youtube(input_text.strip())
            elif input_type == 'url':
                processed_text = extract_text_from_url(input_text.strip())
            elif input_type == 'file':
                st.warning("Please use the file uploader for files.")
                processed_text = ""
            else:
                processed_text = input_text.strip()
            if processed_text:
                summary = summarizer(processed_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                sentiment_result = sentiment(processed_text)[0]
                doc = nlp(processed_text)
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
                # Use default translation (French)
                from transformers import pipeline as hfpipe
                translator = hfpipe("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
                translated = translator(summary)[0]["translation_text"]
                # 4. Results as dismissible glass cards
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                for icon, title, content in [
                    ("<span class='icon-feather'>&#129528;</span>", "Summary", summary),
                    ("<span class='icon-key'>&#128273;</span>", "Sentiment", f"{sentiment_result['label']} <br> <span style='font-size:0.98rem;opacity:0.7;'>Confidence: {round(sentiment_result['score']*100, 2)}%</span>"),
                    ("<span class='icon-lock'>&#128274;</span>", "Entities", ', '.join([f"{e['text']} ({e['label']})" for e in entities]) if entities else "No named entities found."),
                    ("<span class='icon-laurel'>&#127807;</span>", "French Translation", translated)
                ]:
                    st.markdown(f'''<div class="glass-card" style="position:relative;">
                        <div class="result-title">{icon} {title} <span onclick="this.parentElement.parentElement.style.display='none'" style="float:right;cursor:pointer;font-size:1.1em;opacity:0.5;">✖️</span></div>
                        <div class="result-content">{content}</div>
                    </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter something to analyze.")

st.markdown('</div>', unsafe_allow_html=True)
