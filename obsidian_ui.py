import streamlit as st
from transformers import pipeline
import spacy
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from PIL import Image
import pytesseract
import whisper
import openai
import base64

# Setup API key for OpenAI (replace with st.secrets in production)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page Setup
st.set_page_config(page_title="Obsidian Protocol", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>body { background-color: #0E1117; color: #FAFAFA; }</style>""", unsafe_allow_html=True)

# Cache models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    spacy_model = spacy.load("en_core_web_sm")
    whisper_model = whisper.load_model("base")
    return summarizer, classifier, translator, spacy_model, whisper_model

summarizer, classifier, translator, nlp, whisper_model = load_models()

# Title
st.title("🧠 Obsidian Protocol")
st.subheader("Reveal truth in text, image, video, audio or link — powered by AI")

# Input Modes
input_mode = st.radio("Select input type", ["Text", "Upload File", "Article URL", "YouTube Video", "Image", "Audio", "Chat with GPT"])
input_text = ""

# Utilities
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

def extract_text_from_url(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))
    except:
        return "Unable to extract text."

def extract_transcript_from_youtube(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except Exception as e:
        return f"Transcript fetch failed: {str(e)}"

def transcribe_audio(audio_file):
    audio_bytes = audio_file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe("temp.wav")
    return result["text"]

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def download_link(text, filename="obsidian_output.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Result</a>'

# Input Handling
if input_mode == "Text":
    input_text = st.text_area("Paste your text:")

elif input_mode == "Upload File":
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if file:
        input_text = extract_text_from_pdf(file) if file.type == "application/pdf" else file.read().decode()

elif input_mode == "Article URL":
    url = st.text_input("Enter article URL")
    if url:
        input_text = extract_text_from_url(url)

elif input_mode == "YouTube Video":
    yt_url = st.text_input("Enter YouTube video URL")
    if yt_url:
        input_text = extract_transcript_from_youtube(yt_url)

elif input_mode == "Image":
    img_file = st.file_uploader("Upload PNG/JPG image", type=["png", "jpg"])
    if img_file:
        input_text = extract_text_from_image(img_file)

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
    if audio_file:
        input_text = transcribe_audio(audio_file)

elif input_mode == "Chat with GPT":
    chat_prompt = st.text_area("Ask GPT anything:")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": chat_prompt}]
            )
            st.markdown("### 💬 GPT Response")
            st.success(response.choices[0].message.content)

# 🔍 Analysis
if input_mode != "Chat with GPT" and st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please provide valid input.")
    else:
        with st.spinner("Analyzing..."):
            summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
            sentiment = classifier(input_text[:512])[0]
            translated = translator(summary)[0]['translation_text']
            doc = nlp(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Output
        st.markdown("### 🧠 Summary")
        st.info(summary)

        st.markdown("### 🌍 French Translation")
        st.info(translated)

        st.markdown("### 🎭 Sentiment")
        st.success(f"**Label:** {sentiment['label']} | **Confidence:** {round(sentiment['score'], 2)}")

        st.markdown("### 🕵️ Named Entities")
        if entities:
            for ent, label in entities:
                st.write(f"• **{ent}** ({label})")
        else:
            st.write("No named entities found.")

        result = f"Summary:\n{summary}\n\nSentiment: {sentiment['label']} ({round(sentiment['score'],2)})\n\nEntities: {entities}"
        st.markdown(download_link(result), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Obsidian Protocol ✦ All-in-One AI Truth Analyzer ✦ Built with Streamlit + HuggingFace + Whisper + OpenAI")
