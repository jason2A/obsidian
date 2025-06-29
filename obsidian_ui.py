import streamlit as st
from transformers import pipeline
import spacy
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io

# 🧠 Streamlit Config — must be first Streamlit command
st.set_page_config(page_title="Obsidian Protocol", layout="wide", initial_sidebar_state="expanded")

# 📦 Load NLP models
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 🎨 App UI
st.title("🧠 Obsidian Protocol")
st.subheader("Reveal the truth behind any media, speech, or post.")

input_mode = st.radio("Select input type:", ["Text Input", "Upload File", "Article URL", "YouTube Video"])

# 📥 Get text based on input mode
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_url(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)
        return text
    except:
        return "Unable to extract text from URL."

def extract_transcript_from_youtube(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except Exception as e:
        return f"Transcript fetch failed: {str(e)}"

input_text = ""

if input_mode == "Text Input":
    input_text = st.text_area("Paste your article, speech, or social post:")

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload .txt or .pdf file", type=["txt", "pdf"])
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
    yt_url = st.text_input("Enter YouTube video link:")
    if yt_url:
        input_text = extract_transcript_from_youtube(yt_url)

# 🔍 Analyze Button
if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please provide valid input.")
    else:
        with st.spinner("Analyzing..."):
            # Summary
            summary = summarizer(input_text, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]

            # Sentiment
            sentiment = classifier(input_text[:512])[0]

            # Entities
            doc = nlp(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Output
        st.markdown("### 🧠 Summary")
        st.info(summary)

        st.markdown("### 🎭 Sentiment")
        st.success(f"**Label:** {sentiment['label']} &nbsp;&nbsp; | &nbsp;&nbsp; **Confidence:** {round(sentiment['score'], 2)}")

        st.markdown("### 🕵️ Named Entities")
        if entities:
            for entity, label in entities:
                st.write(f"• **{entity}** ({label})")
        else:
            st.write("No named entities found.")
