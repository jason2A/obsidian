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

# Main CSS and all the content will go here...
# Interactive background JS/CSS - FIXED VERSION
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
    // Add aurora waves - FIXED: Removed problematic JavaScript variable interpolation
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
    // Add bokeh circles - FIXED: Similar fix applied
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

st.write("✅ Fixed version - JavaScript variables no longer cause NameError!")