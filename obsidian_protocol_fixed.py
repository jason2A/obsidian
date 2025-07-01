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