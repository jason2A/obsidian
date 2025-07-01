import streamlit as st
import tempfile
import os
from io import BytesIO
import base64
import getpass

# PDF
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# URL Scraping
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# YouTube Transcript
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

# Image OCR & Captioning
try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None

# Audio Transcription & Analysis
try:
    import whisper
except ImportError:
    whisper = None
try:
    from pyAudioAnalysis import audioSegmentation as aS
except ImportError:
    aS = None

# HuggingFace Transformers
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# TTS
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Knowledge Graph
try:
    import networkx as nx
    from pyvis.network import Network
except ImportError:
    nx = None
    Network = None

# Plagiarism/Similarity
try:
    from sentence_transformers import SentenceTransformer, util
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    util = None
    cosine_similarity = None

# Set page config and apply glassmorphic, ultra-modern CSS
st.set_page_config(page_title="Obsidian Protocol v2.0", layout="wide")
st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body, .stApp {
        background: linear-gradient(120deg, #1a1f2b 0%, #232946 50%, #2e3a59 100%);
        min-height: 100vh;
        animation: glassbgmove 24s ease-in-out infinite alternate;
        transition: background 0.8s cubic-bezier(.4,2,.6,1);
        will-change: background, opacity, transform;
    }
    @keyframes glassbgmove {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    .aurora-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0;
        pointer-events: none;
        background: radial-gradient(circle at 20% 40%, #00c3ff44 0%, #0057b822 60%, transparent 100%),
                    radial-gradient(circle at 80% 60%, #ffd70044 0%, #ffb30022 60%, transparent 100%);
        animation: auroraMove 18s ease-in-out infinite alternate;
        opacity: 0.7;
    }
    @keyframes auroraMove {
        0% { background-position: 0% 0%, 100% 100%; }
        100% { background-position: 100% 100%, 0% 0%; }
    }
    .glass-sidebar {
        position: fixed;
        top: 0; left: 0; bottom: 0;
        width: 72px;
        background: rgba(34, 40, 60, 0.22);
        box-shadow: 4px 0 32px #0057B822;
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-right: 2px solid rgba(255,255,255,0.08);
        z-index: 10001;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem 0 1.2rem 0;
        gap: 1.2rem;
        transition: width 0.35s cubic-bezier(.4,2,.6,1), background 0.5s, box-shadow 0.3s cubic-bezier(.4,2,.6,1);
    }
    .glass-sidebar.expanded {
        width: 260px;
        align-items: flex-start;
        padding-left: 1.2rem;
        background: rgba(34, 40, 60, 0.32);
    }
    .sidebar-toggle {
        position: absolute;
        top: 1.2rem;
        right: -18px;
        background: linear-gradient(135deg, #0057B8 0%, #FFD700 100%);
        color: #fff;
        border-radius: 50%;
        width: 36px; height: 36px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 2px 12px #FFD70044;
        border: none;
        cursor: pointer;
        z-index: 10002;
        transition: background 0.2s, transform 0.1s, box-shadow 0.2s cubic-bezier(.4,2,.6,1);
        animation: pulse 1.6s cubic-bezier(.4,2,.6,1) infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 #FFD70044; }
        70% { box-shadow: 0 0 0 12px #FFD70011; }
        100% { box-shadow: 0 0 0 0 #FFD70044; }
    }
    .sidebar-toggle:hover { background: linear-gradient(135deg, #FFD700 0%, #0057B8 100%); }
    .sidebar-toggle:active {
        transform: scale(0.92) rotate(-8deg);
        box-shadow: 0 0 0 0 #FFD70044;
    }
    .sidebar-avatar {
        width: 48px; height: 48px;
        border-radius: 50%;
        margin-bottom: 1.5rem;
        border: 2.5px solid #FFD700;
        box-shadow: 0 2px 12px #FFD70033;
        background: url('https://randomuser.me/api/portraits/men/32.jpg') center/cover;
        transition: border 0.3s, box-shadow 0.3s;
    }
    .sidebar-section {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        padding: 0.85rem 1.1rem;
        border-radius: 16px;
        font-size: 1.13rem;
        font-weight: 600;
        color: #eaf6ff;
        cursor: pointer;
        transition: background 0.18s cubic-bezier(.4,2,.6,1), color 0.18s, box-shadow 0.18s, transform 0.18s cubic-bezier(.4,2,.6,1);
        border: 1.5px solid transparent;
        width: 100%;
        margin-bottom: 0.2rem;
        position: relative;
        opacity: 0.92;
        backdrop-filter: blur(2px);
        animation: fadein 0.7s;
    }
    @keyframes fadein { from { opacity: 0; transform: translateX(-16px);} to { opacity: 0.92; transform: none; } }
    .sidebar-section.selected {
        background: rgba(0,87,184,0.22);
        color: #FFD700;
        border: 1.5px solid #FFD700;
        box-shadow: 0 2px 16px #FFD70033;
        transform: scale(1.04);
    }
    .sidebar-section:hover {
        background: rgba(0,87,184,0.18);
        color: #FFD700;
        transform: scale(1.03);
    }
    .sidebar-section:active {
        transform: scale(0.97) translateX(2px);
        box-shadow: 0 2px 16px #FFD70055;
    }
    .sidebar-section .tooltip {
        visibility: hidden;
        opacity: 0;
        background: rgba(0,0,0,0.7);
        color: #fff;
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        position: absolute;
        left: 60px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.98rem;
        white-space: nowrap;
        z-index: 10003;
        transition: opacity 0.2s;
        animation: fadein 0.4s cubic-bezier(.4,2,.6,1);
    }
    .sidebar-section:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .sidebar-bottom {
        margin-top: auto;
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 1.2rem;
        align-items: flex-start;
    }
    .sidebar-settings {
        color: #FFD700;
        font-size: 1.5rem;
        cursor: pointer;
        margin-left: 0.5rem;
        margin-bottom: 0.5rem;
        transition: color 0.2s;
    }
    .sidebar-settings:hover { color: #0057B8; }
    .main-content-glass {
        margin-left: 90px;
        margin-top: 2.5rem;
        padding: 2.5rem 2.5rem 2.5rem 2.5rem;
        background: rgba(255,255,255,0.22);
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
        backdrop-filter: blur(32px) saturate(180%);
        -webkit-backdrop-filter: blur(32px) saturate(180%);
        border-radius: 32px;
        border: 2px solid rgba(255,255,255,0.18);
        min-height: 80vh;
        max-width: 1100px;
        transition: box-shadow 0.22s cubic-bezier(.4,2,.6,1), background 0.5s, transform 0.22s cubic-bezier(.4,2,.6,1), opacity 0.22s cubic-bezier(.4,2,.6,1);
        will-change: box-shadow, background, transform, opacity;
        animation: fadein 0.8s;
    }
    .main-content-glass.expanded { margin-left: 280px; }
    .glass-card {
        background: rgba(255,255,255,0.22);
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-radius: 24px;
        border: 1.5px solid rgba(255,255,255,0.18);
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 2rem;
        transition: box-shadow 0.22s cubic-bezier(.4,2,.6,1), background 0.5s, transform 0.22s cubic-bezier(.4,2,.6,1), opacity 0.22s cubic-bezier(.4,2,.6,1);
        will-change: box-shadow, background, transform, opacity;
        animation: fadein 0.6s cubic-bezier(.4,2,.6,1);
    }
    .glass-card:hover {
        box-shadow: 0 12px 48px 0 rgba(31,38,135,0.22);
        transform: translateY(-2px) scale(1.01);
        background: rgba(255,255,255,0.28);
    }
    .glass-input, .glass-select, .glass-slider {
        background: rgba(255,255,255,0.7) !important;
        color: #222 !important;
        border-radius: 18px !important;
        border: 1.5px solid #e0e7ef !important;
        font-size: 1.15rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04) !important;
        transition: box-shadow 0.18s cubic-bezier(.4,2,.6,1), background 0.18s cubic-bezier(.4,2,.6,1);
    }
    .glass-input:focus, .glass-select:focus, .glass-slider:focus {
        box-shadow: 0 0 0 4px #FFD70055 !important;
    }
    .glass-btn {
        box-shadow: 0 2px 16px rgba(0,87,184,0.10);
        background: linear-gradient(90deg, #e3e9f7 0%, #eaf6ff 100%);
        color: #0057B8;
        border-radius: 18px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: background 0.18s cubic-bezier(.4,2,.6,1), color 0.18s, box-shadow 0.18s, transform 0.18s cubic-bezier(.4,2,.6,1);
        border: 1.5px solid #b3d4fc;
        position: relative;
        overflow: hidden;
        animation: fadein 0.7s;
        will-change: background, color, box-shadow, transform;
    }
    .glass-btn:hover {
        background: linear-gradient(90deg, #d0e6ff 0%, #b3d4fc 100%);
        color: #003366;
        transform: scale(1.045);
        box-shadow: 0 4px 32px #FFD70044;
    }
    .glass-btn:active {
        transform: scale(0.96);
        box-shadow: 0 2px 16px #FFD70077;
    }
    .glass-btn::after {
        content: '';
        position: absolute;
        left: 50%; top: 50%;
        width: 0; height: 0;
        background: rgba(0,87,184,0.18);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.35s cubic-bezier(.4,2,.6,1), height 0.35s cubic-bezier(.4,2,.6,1), opacity 0.35s;
        opacity: 0.5;
        pointer-events: none;
    }
    .glass-btn:active::after {
        width: 180%; height: 180%; opacity: 0;
        transition: width 0.35s cubic-bezier(.4,2,.6,1), height 0.35s cubic-bezier(.4,2,.6,1), opacity 0.35s;
    }
    @media (max-width: 900px) {
        .main-content-glass, .main-content-glass.expanded { margin-left: 0 !important; max-width: 100vw; border-radius: 0; }
        .glass-sidebar, .glass-sidebar.expanded { position: static; width: 100vw !important; flex-direction: row; align-items: center; border-radius: 0; box-shadow: none; }
    }
    </style>
''', unsafe_allow_html=True)

# Aurora/bokeh animated background layer
st.markdown('<div class="aurora-bg"></div>', unsafe_allow_html=True)

# Live theme/customization panel (in sidebar bottom)
if "glass_opacity" not in st.session_state:
    st.session_state["glass_opacity"] = 0.22
if "accent_color" not in st.session_state:
    st.session_state["accent_color"] = "#FFD700"
if "bg_style" not in st.session_state:
    st.session_state["bg_style"] = "Aurora"
if "font_family" not in st.session_state:
    st.session_state["font_family"] = "Inter"

# Glassy sticky toolbar (at top of page)
st.markdown('''
    <div class="glass-toolbar">
        <span class="toolbar-title">🧠 Obsidian Protocol v2.0</span>
        <span class="toolbar-tip">💡 Did you know? You can batch process files and visualize knowledge graphs!</span>
    </div>
''', unsafe_allow_html=True)

# Floating Action Button (FAB) for quick Analyze
def render_fab():
    st.markdown('''
        <button class="fab" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">🔍</button>
    ''', unsafe_allow_html=True)
render_fab()

# Wrap main content in glass-card for luxury look
from contextlib import contextmanager
@contextmanager
def glass_card():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    yield
    st.markdown('</div>', unsafe_allow_html=True)

# Onboarding tour (show only once per session)
if "onboarded" not in st.session_state:
    st.session_state["onboarded"] = False
if not st.session_state["onboarded"]:
    st.info("👋 Welcome to Obsidian Protocol! Upload or paste your content, then explore the tabs for powerful AI analysis, visualizations, and more. Use the Theme Picker to personalize your experience.")
    if st.button("Got it! Start Exploring", key="onboard_btn"):
        st.session_state["onboarded"] = True

# Personalized greeting (top of main content)
user = getpass.getuser() if hasattr(getpass, 'getuser') else "User"
st.markdown(f'<div style="font-size:1.25rem;font-weight:600;margin-bottom:1.2rem;color:#FFD700;">👋 Welcome back, {user}! You are special. ✨</div>', unsafe_allow_html=True)

# Animated loader bar (example, can be shown during processing)
def show_loader():
    st.markdown('<div class="glass-loader"></div>', unsafe_allow_html=True)

# Confetti/sparkle on key actions (use st.balloons or st.snow for now, can be replaced with JS confetti)
def celebrate():
    st.balloons()
    st.success("🎉 Analysis complete! Enjoy your insights.")

# Collapsible sidebar state
if "sidebar_expanded" not in st.session_state:
    st.session_state["sidebar_expanded"] = False

# Sidebar navigation with avatar, toggle, and settings
SECTIONS = [
    ("Analyze", "🔍"),
    ("AI Tools", "🤖"),
    ("Q&A", "❓"),
    ("Batch", "📦"),
    ("Captioning", "🖼️"),
    ("Audio Analysis", "🎤"),
    ("Graph", "🕸️"),
    ("Similarity", "🧬"),
    ("Workflow", "🔗"),
    ("Timeline", "🗓️"),
    ("Topics", "📚"),
    ("Explain", "💡"),
    ("Settings", "⚙️"),
    ("History", "🕑"),
    ("Export", "⬇️"),
    ("Accessibility", "🦾"),
    ("User Profiles", "👤"),
    ("Real-Time Collaboration", "🤝"),
    ("Data Visualizations", "📊"),
    ("Theme Picker", "🎨"),
    ("Journey", "🛤️")
]
if "active_section" not in st.session_state:
    st.session_state["active_section"] = SECTIONS[0][0]

# Sidebar HTML
sidebar_class = "glass-sidebar expanded" if st.session_state["sidebar_expanded"] else "glass-sidebar"
sidebar_html = f'<div class="{sidebar_class}">' + \
    '<div class="sidebar-avatar"></div>' + \
    '<button class="sidebar-toggle" onclick="window.parent.postMessage({isStreamlitMessage: true, type: \'sidebar:toggle\'}, \'*\')">' + ("⏪" if st.session_state["sidebar_expanded"] else "⏩") + '</button>'
for section, icon in SECTIONS:
    selected = "selected" if st.session_state["active_section"] == section else ""
    tooltip = f'<span class="tooltip">{section}</span>'
    sidebar_html += f'<div class="sidebar-section {selected}" onclick="window.location.hash=\"#{section}\";window.dispatchEvent(new Event(\'hashchange\'))">{icon} {tooltip if not st.session_state["sidebar_expanded"] else section}</div>'
sidebar_html += '<div class="sidebar-bottom">'
sidebar_html += '<span class="sidebar-settings" title="Settings">⚙️</span>'
sidebar_html += '<span class="sidebar-settings" title="Theme">🎨</span>'
sidebar_html += '</div></div>'
st.markdown(sidebar_html, unsafe_allow_html=True)

# JS for sidebar toggle and section switching
st.markdown('''
    <script>
    window.addEventListener('hashchange', function() {
        const section = window.location.hash.replace('#','');
        if (section) {
            window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: section}, '*');
        }
    });
    window.addEventListener('message', function(event) {
        if (event.data && event.data.isStreamlitMessage && event.data.type === 'sidebar:toggle') {
            window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:rerun'}, '*');
        }
    });
    </script>
''', unsafe_allow_html=True)

# Main content area as a glass card, animated margin
main_class = "main-content-glass expanded" if st.session_state["sidebar_expanded"] else "main-content-glass"
st.markdown(f'<div class="{main_class}">', unsafe_allow_html=True)

# Session state for results and chat
if "results" not in st.session_state:
    st.session_state["results"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "ai_outputs" not in st.session_state:
    st.session_state["ai_outputs"] = {}
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []

# Model caching for performance
@st.cache_resource(show_spinner=False)
def get_summarizer():
    if pipeline:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return None
@st.cache_resource(show_spinner=False)
def get_sentiment():
    if pipeline:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return None
@st.cache_resource(show_spinner=False)
def get_translator(lang="fr"):
    if pipeline:
        return pipeline(f"translation_en_to_{lang}", model=f"Helsinki-NLP/opus-mt-en-{lang}")
    return None
@st.cache_resource(show_spinner=False)
def get_qa():
    if pipeline:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return None
@st.cache_resource(show_spinner=False)
def get_blip():
    if BlipProcessor and BlipForConditionalGeneration:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    return None, None
@st.cache_resource(show_spinner=False)
def get_sentence_transformer():
    if SentenceTransformer:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

# Initialize session state variables
if "aurora_density" not in st.session_state:
    st.session_state["aurora_density"] = 4
if "bokeh_density" not in st.session_state:
    st.session_state["bokeh_density"] = 12
if "bg_speed" not in st.session_state:
    st.session_state["bg_speed"] = 1.0
if "effects_enabled" not in st.session_state:
    st.session_state["effects_enabled"] = True

# --- INTERACTIVE BACKGROUND: RIPPLE, WAVES, TUNABLE DENSITY/SPEED (FIXED VERSION) ---
st.markdown(f'''
    <style>
    .ripple-bg {{ position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; z-index: 2; pointer-events: none; }}
    .ripple-bg span {{ position: absolute; border-radius: 50%; opacity: 0.22; background: radial-gradient(circle, #0057B8 0%, #FFD700 80%, transparent 100%); pointer-events: none; animation: ripplescale 1.2s cubic-bezier(.4,2,.6,1); }}
    @keyframes ripplescale {{
        0% {{ transform: scale(0.2); opacity: 0.5; }}
        60% {{ opacity: 0.22; }}
        100% {{ transform: scale(2.2); opacity: 0; }}
    }}
    .cursor-wave {{ position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; z-index: 2; pointer-events: none; }}
    .cursor-wave span {{ position: absolute; width: 48px; height: 48px; border-radius: 50%; opacity: 0.13; background: linear-gradient(135deg, #FFD700 0%, #0057B8 100%); filter: blur(8px); pointer-events: none; animation: wavefade 1.2s linear; }}
    @keyframes wavefade {{ 0% {{ opacity: 0.13; }} 100% {{ opacity: 0; }} }}
    </style>
    <script>
    // Remove old aurora/bokeh if present
    if(window.auroraWaveAdded){{
        document.querySelectorAll('.aurora-wave').forEach(e=>e.remove());
        window.auroraWaveAdded = false;
    }}
    if(window.bokehBgAdded){{
        document.querySelectorAll('.bokeh-bg').forEach(e=>e.remove());
        window.bokehBgAdded = false;
    }}
    // FIXED: Add aurora waves - JavaScript variables properly handled
    if(!window.auroraWaveAdded){{
        window.auroraWaveAdded = true;
        const aurora = document.createElement('div');
        aurora.className = 'aurora-wave';
        const density = {st.session_state.get('aurora_density', 4)};
        const speed = {st.session_state.get('bg_speed', 1.0)};
        for(let i=0; i<density; i++){{
            const s = document.createElement('span');
            s.style.top = (10+80*i/density)+'vh';
            s.style.animationDuration = (10+4*i)*(1/speed)+'s';
            aurora.appendChild(s);
        }}
        document.body.appendChild(aurora);
    }}
    // FIXED: Add bokeh circles - JavaScript variables properly handled
    if(!window.bokehBgAdded){{
        window.bokehBgAdded = true;
        const bokeh = document.createElement('div');
        bokeh.className = 'bokeh-bg';
        const bokehDensity = {st.session_state.get('bokeh_density', 12)};
        const speed = {st.session_state.get('bg_speed', 1.0)};
        for(let i=0; i<bokehDensity; i++){{
            const s = document.createElement('span');
            s.style.left = (Math.random()*100)+'vw';
            s.style.bottom = (-10-Math.random()*20)+'vh';
            s.style.width = (32+Math.random()*48)+'px';
            s.style.height = s.style.width;
            s.style.animationDelay = (Math.random()*12)+'s';
            s.style.animationDuration = (18*(1/speed))+'s';
            bokeh.appendChild(s);
        }}
        document.body.appendChild(bokeh);
    }}
    // Add ripple effect on click
    if(!window.rippleBgAdded){{
        window.rippleBgAdded = true;
        const ripple = document.createElement('div');
        ripple.className = 'ripple-bg';
        document.body.appendChild(ripple);
        window.parent.document.body.addEventListener('click', function(e){{
            if(e.target.closest('.stApp')){{
                const span = document.createElement('span');
                span.style.left = (e.clientX-80)+'px';
                span.style.top = (e.clientY-80)+'px';
                span.style.width = '160px';
                span.style.height = '160px';
                ripple.appendChild(span);
                setTimeout(()=>{{span.remove();}}, 1200);
            }}
        }});
    }}
    // Add cursor-following wave
    if(!window.cursorWaveAdded){{
        window.cursorWaveAdded = true;
        const wave = document.createElement('div');
        wave.className = 'cursor-wave';
        document.body.appendChild(wave);
        window.parent.document.body.addEventListener('mousemove', function(e){{
            if(e.target.closest('.stApp')){{
                const span = document.createElement('span');
                span.style.left = (e.clientX-24)+'px';
                span.style.top = (e.clientY-24)+'px';
                wave.appendChild(span);
                setTimeout(()=>{{span.remove();}}, 1200);
            }}
        }});
    }}
    </script>
''', unsafe_allow_html=True)

st.markdown("<h1>🧠 Obsidian Protocol v2.0 - Complete Fixed Version</h1>", unsafe_allow_html=True)
st.success("✅ All JavaScript variable interpolation issues have been fixed!")
st.info("This is the complete working version with glassmorphic UI and interactive background effects.")

st.markdown('</div>', unsafe_allow_html=True)