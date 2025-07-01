import streamlit as st
import tempfile
import os
from io import BytesIO
import base64

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

st.set_page_config(page_title="Obsidian Protocol v2.0", layout="wide")

# Now you can use st.markdown, st.sidebar, etc.
st.markdown('''
    <style>
    body, .stApp { background-color: #000014; color: #E4E6EB; }
    .stTextInput>div>div>input, .stTextArea textarea, .stButton>button {
        background: #0A0A23; color: #E4E6EB; border-radius: 8px; border: none;
    }
    .stTabs [data-baseweb="tab"] { background: #0A0A23; color: #E4E6EB; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { background: #001F3F; color: #FFD700; }
    .stCodeBlock { background: #0A0A23 !important; }
    .stSidebar { background: #001F3F !important; }
    </style>
''', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 Obsidian Protocol v2.0")
st.sidebar.markdown("World-Changing AI Media Analyzer")
st.sidebar.info("Upload, analyze, and transform any media. All local, all free.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Features:**\n- Multilingual, TTS, Q&A, Captioning, Batch, Graphs, Privacy\n- No OpenAI API required\n- Free forever (local models/public APIs)")

# Tabs for main features
TABS = [
    "Analyze", "AI Tools", "Q&A", "Batch", "Captioning", "Audio Analysis", "Graph", "Similarity", "Workflow", "Timeline", "Topics", "Explain", "Settings", "History", "Export", "Accessibility", "User Profiles", "Real-Time Collaboration", "Data Visualizations", "Theme Picker"
]
tabs = st.tabs(TABS)

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

# --- Analyze Tab ---
with tabs[0]:
    st.header("Analyze File or Content")
    analyze_type = st.selectbox("Select input type:", ["Text", "TXT File", "PDF", "URL", "YouTube Video", "Image", "Audio"])
    content = ""
    lang = st.selectbox("Select language for translation/output:", ["en", "fr", "es", "de", "zh", "ar", "ru", "hi", "pt", "it"])
    if analyze_type == "Text":
        content = st.text_area("Enter text to analyze:")
    elif analyze_type == "TXT File":
        txt_file = st.file_uploader("Upload a TXT file", type=["txt"])
        if txt_file:
            content = txt_file.read().decode("utf-8")
            st.code(content, language="text")
    elif analyze_type == "PDF":
        if PyPDF2:
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                content = "\n".join([page.extract_text() or "" for page in reader.pages])
                st.code(content, language="text")
        else:
            st.warning("PyPDF2 not installed. Run: pip install PyPDF2")
    elif analyze_type == "URL":
        if requests and BeautifulSoup:
            url = st.text_input("Enter a URL to scrape article text:")
            if url:
                try:
                    r = requests.get(url)
                    soup = BeautifulSoup(r.text, "html.parser")
                    paragraphs = [p.get_text() for p in soup.find_all("p")]
                    content = "\n".join(paragraphs)
                    st.code(content, language="text")
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
        else:
            st.warning("requests and beautifulsoup4 not installed. Run: pip install requests beautifulsoup4")
    elif analyze_type == "YouTube Video":
        if YouTubeTranscriptApi:
            yt_url = st.text_input("Enter YouTube video URL:")
            if yt_url:
                import re
                match = re.search(r"v=([\w-]+)", yt_url)
                if match:
                    video_id = match.group(1)
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        content = " ".join([x['text'] for x in transcript])
                        st.code(content, language="text")
                    except Exception as e:
                        st.error(f"Could not fetch transcript: {e}")
                else:
                    st.warning("Invalid YouTube URL.")
        else:
            st.warning("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
    elif analyze_type == "Image":
        if Image and pytesseract:
            img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if img_file:
                img = Image.open(img_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                content = pytesseract.image_to_string(img)
                st.code(content, language="text")
        else:
            st.warning("Pillow and pytesseract not installed. Run: pip install pillow pytesseract")
    elif analyze_type == "Audio":
        if whisper:
            audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                model = whisper.load_model("base")
                result = model.transcribe(tmp_path)
                content = result["text"]
                st.code(content, language="text")
                os.remove(tmp_path)
        else:
            st.warning("openai-whisper not installed. Run: pip install openai-whisper")
    # Store content for further analysis
    st.session_state["results"] = content
    st.markdown("---")
    st.subheader("AI Tools (see next tab)")
    if content:
        st.download_button("Download Results as .txt", content, file_name="obsidian_output.txt")
        # TTS
        st.markdown("**Text-to-Speech (TTS):**")
        tts_engine = st.radio("TTS Engine", ["gTTS (Google)", "pyttsx3 (Offline)"])
        if st.button("Speak/Download Audio"):
            if tts_engine == "gTTS (Google)" and gTTS:
                tts = gTTS(content, lang=lang)
                tts_fp = BytesIO()
                tts.write_to_fp(tts_fp)
                tts_fp.seek(0)
                b64 = base64.b64encode(tts_fp.read()).decode()
                st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")
                st.download_button("Download Audio", data=tts_fp, file_name="tts.mp3")
            elif tts_engine == "pyttsx3 (Offline)" and pyttsx3:
                engine = pyttsx3.init()
                engine.save_to_file(content, "tts.wav")
                engine.runAndWait()
                with open("tts.wav", "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                st.download_button("Download Audio", data=open("tts.wav", "rb"), file_name="tts.wav")
                os.remove("tts.wav")
            else:
                st.warning("TTS engine not installed. Run: pip install gtts pyttsx3")

# --- AI Tools Tab ---
with tabs[1]:
    st.header("AI Tools")
    content = st.session_state["results"]
    if not content:
        st.info("No content loaded. Please use the Analyze tab first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summarization")
            summarizer = get_summarizer()
            if summarizer:
                if st.button("Summarize", key="summarize_btn") or st.session_state["ai_outputs"].get("summary"):
                    if not st.session_state["ai_outputs"].get("summary"):
                        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                        st.session_state["ai_outputs"]["summary"] = summary
                    st.success(st.session_state["ai_outputs"]["summary"])
            else:
                st.warning("Transformers not installed. Run: pip install transformers torch")
        with col2:
            st.subheader("Sentiment Analysis")
            sentiment = get_sentiment()
            if sentiment:
                if st.button("Analyze Sentiment", key="sentiment_btn") or st.session_state["ai_outputs"].get("sentiment"):
                    if not st.session_state["ai_outputs"].get("sentiment"):
                        sent = sentiment(content)[0]
                        st.session_state["ai_outputs"]["sentiment"] = f"{sent['label']} (score: {sent['score']:.2f})"
                    st.success(st.session_state["ai_outputs"]["sentiment"])
            else:
                st.warning("Transformers not installed. Run: pip install transformers torch")
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Named Entity Recognition (NER)")
            if nlp:
                if st.button("Extract Entities", key="ner_btn") or st.session_state["ai_outputs"].get("ner"):
                    if not st.session_state["ai_outputs"].get("ner"):
                        doc = nlp(content)
                        ents = [(ent.text, ent.label_) for ent in doc.ents]
                        st.session_state["ai_outputs"]["ner"] = ents
                    ents = st.session_state["ai_outputs"]["ner"]
                    if ents:
                        for ent, label in ents:
                            st.markdown(f"- **{ent}**: {label}")
                    else:
                        st.info("No named entities found.")
            else:
                st.warning("spaCy not installed or model missing. Run: pip install spacy && python -m spacy download en_core_web_sm")
        with col4:
            st.subheader(f"Translation (EN → {lang.upper()})")
            try:
                translator = get_translator(lang)
                translation_error = None
            except Exception as e:
                translator = None
                translation_error = str(e)
            if translator:
                if st.button("Translate", key="translate_btn") or st.session_state["ai_outputs"].get("translation"):
                    if not st.session_state["ai_outputs"].get("translation"):
                        try:
                            translation = translator(content)[0]['translation_text']
                            st.session_state["ai_outputs"]["translation"] = translation
                        except Exception as e:
                            st.session_state["ai_outputs"]["translation"] = f"[Translation Error]: {e}"
                    st.success(st.session_state["ai_outputs"]["translation"])
            else:
                if translation_error:
                    st.warning(f"Translation model could not be loaded: {translation_error}")
                else:
                    st.warning("Transformers not installed. Run: pip install transformers torch")

        # --- Analyze All Button ---
        st.markdown("---")
        if st.button("Analyze All", key="analyze_all_btn"):
            # Summarization
            if summarizer:
                try:
                    summary = summarizer(content, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                except Exception as e:
                    summary = f"[Summarization Error]: {e}"
            else:
                summary = "[Summarizer not available]"
            st.session_state["ai_outputs"]["summary"] = summary
            # Sentiment
            if sentiment:
                try:
                    sent = sentiment(content)[0]
                    sentiment_result = f"{sent['label']} (score: {sent['score']:.2f})"
                except Exception as e:
                    sentiment_result = f"[Sentiment Error]: {e}"
            else:
                sentiment_result = "[Sentiment model not available]"
            st.session_state["ai_outputs"]["sentiment"] = sentiment_result
            # NER
            if nlp:
                try:
                    doc = nlp(content)
                    ents = [(ent.text, ent.label_) for ent in doc.ents]
                except Exception as e:
                    ents = [(f"[NER Error]: {e}", "ERROR")]
            else:
                ents = []
            st.session_state["ai_outputs"]["ner"] = ents
            # Translation
            if translator:
                try:
                    translation = translator(content)[0]['translation_text']
                except Exception as e:
                    translation = f"[Translation Error]: {e}"
            else:
                translation = "[Translation model not available]"
            st.session_state["ai_outputs"]["translation"] = translation

        # --- Display All Results if available ---
        if st.session_state["ai_outputs"].get("summary"):
            st.markdown("**Summary:**")
            st.success(st.session_state["ai_outputs"]["summary"])
        if st.session_state["ai_outputs"].get("sentiment"):
            st.markdown("**Sentiment:**")
            st.success(st.session_state["ai_outputs"]["sentiment"])
        if st.session_state["ai_outputs"].get("ner"):
            st.markdown("**Named Entities:**")
            ents = st.session_state["ai_outputs"]["ner"]
            if ents:
                for ent, label in ents:
                    st.markdown(f"- **{ent}**: {label}")
            else:
                st.info("No named entities found.")
        if st.session_state["ai_outputs"].get("translation"):
            st.markdown(f"**Translation (EN → {lang.upper()}):**")
            st.success(st.session_state["ai_outputs"]["translation"])

# --- Q&A Tab ---
with tabs[2]:
    st.header("Document Q&A")
    content = st.session_state["results"]
    if not content:
        st.info("No content loaded. Please use the Analyze tab first.")
    else:
        qa = get_qa()
        question = st.text_input("Ask a question about the document:")
        if st.button("Get Answer") and question and qa:
            answer = qa(question=question, context=content)
            st.success(f"**Answer:** {answer['answer']}")

# --- Batch Tab ---
with tabs[3]:
    st.header("Batch Processing")
    batch_files = st.file_uploader("Upload multiple files (TXT, PDF, Image, Audio)", type=["txt", "pdf", "jpg", "jpeg", "png", "mp3", "wav", "m4a"], accept_multiple_files=True)
    batch_results = []
    if batch_files:
        for f in batch_files:
            ext = os.path.splitext(f.name)[1].lower()
            text = ""
            if ext == ".txt":
                text = f.read().decode("utf-8")
            elif ext == ".pdf" and PyPDF2:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext in [".jpg", ".jpeg", ".png"] and Image and pytesseract:
                img = Image.open(f)
                text = pytesseract.image_to_string(img)
            elif ext in [".mp3", ".wav", ".m4a"] and whisper:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(f.read())
                    tmp_path = tmp_file.name
                model = whisper.load_model("base")
                result = model.transcribe(tmp_path)
                text = result["text"]
                os.remove(tmp_path)
            batch_results.append((f.name, text))
        st.session_state["batch_results"] = batch_results
        for fname, text in batch_results:
            st.markdown(f"**{fname}:**")
            st.code(text, language="text")
        if batch_results:
            all_text = "\n\n".join([f"{fname}:\n{text}" for fname, text in batch_results])
            st.download_button("Download All Results as .txt", all_text, file_name="obsidian_batch_output.txt")

# --- Captioning Tab ---
with tabs[4]:
    st.header("Image Captioning")
    if BlipProcessor and BlipForConditionalGeneration:
        img_file = st.file_uploader("Upload an image for captioning", type=["jpg", "jpeg", "png"], key="caption_img")
        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            processor, model = get_blip()
            import torch
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.success(f"**Caption:** {caption}")
    else:
        st.warning("BLIP not installed. Run: pip install transformers torch")

# --- Audio Analysis Tab ---
with tabs[5]:
    st.header("Audio Sentiment & Speaker Diarization")
    audio_file = st.file_uploader("Upload audio for analysis", type=["mp3", "wav", "m4a"], key="audio_analysis")
    if audio_file and aS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        st.info("Speaker diarization and audio sentiment coming soon (requires pyAudioAnalysis)")
        # Example: diarization = aS.speaker_diarization(tmp_path, n_speakers=2)
        os.remove(tmp_path)
    elif audio_file:
        st.warning("pyAudioAnalysis not installed. Run: pip install pyAudioAnalysis")

# --- Knowledge Graph Tab ---
with tabs[6]:
    st.header("Interactive Knowledge Graph Extraction")
    content = st.session_state["results"]
    if not content:
        st.info("No content loaded. Please use the Analyze tab first.")
    elif nx and Network and nlp:
        doc = nlp(content)
        G = nx.Graph()
        for ent in doc.ents:
            G.add_node(ent.text, label=ent.label_)
        for i in range(len(doc.ents)-1):
            G.add_edge(doc.ents[i].text, doc.ents[i+1].text)
        net = Network(height="400px", width="100%", bgcolor="#222", font_color="#FFD700")
        net.from_nx(G)
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=450)
        os.remove("graph.html")
        # Clickable entities (show sentences containing entity)
        st.markdown("**Click an entity below to see related sentences:**")
        selected_entity = st.selectbox("Entities:", [ent.text for ent in doc.ents]) if doc.ents else None
        if selected_entity:
            sentences = [sent.text for sent in doc.sents if selected_entity in sent.text]
            st.markdown(f"**Sentences mentioning {selected_entity}:**")
            for s in sentences:
                st.write(s)
    else:
        st.warning("networkx, pyvis, or spaCy not installed. Run: pip install networkx pyvis spacy && python -m spacy download en_core_web_sm")

# --- Similarity/Plagiarism Tab ---
with tabs[7]:
    st.header("Plagiarism & Similarity Detection")
    content = st.session_state["results"]
    if not content:
        st.info("No content loaded. Please use the Analyze tab first.")
    elif SentenceTransformer and cosine_similarity:
        st.markdown("Compare your text to another sample:")
        compare_text = st.text_area("Paste text to compare:")
        if st.button("Check Similarity") and compare_text:
            model = get_sentence_transformer()
            emb1 = model.encode([content])[0].reshape(1, -1)
            emb2 = model.encode([compare_text])[0].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            st.success(f"Cosine Similarity: {sim:.2f}")
            if sim > 0.8:
                st.warning("High similarity detected! Possible plagiarism.")
    else:
        st.warning("sentence-transformers or scikit-learn not installed. Run: pip install sentence-transformers scikit-learn")

# --- Workflow Tab ---
with tabs[8]:
    st.header("Customizable Workflows (Chain Tools)")
    st.info("Select a sequence of tools to apply. (Drag-and-drop coming soon!)")
    steps = st.multiselect("Choose steps:", ["OCR", "Translate", "Summarize", "NER", "TTS"])
    content = st.session_state["results"]
    workflow_result = content
    if st.button("Run Workflow") and content and steps:
        for step in steps:
            if step == "OCR" and Image and pytesseract:
                st.info("OCR already applied if image uploaded.")
            elif step == "Translate" and pipeline:
                translator = get_translator(lang)
                workflow_result = translator(workflow_result)[0]['translation_text']
            elif step == "Summarize" and pipeline:
                summarizer = get_summarizer()
                workflow_result = summarizer(workflow_result, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
            elif step == "NER" and nlp:
                doc = nlp(workflow_result)
                ents = [(ent.text, ent.label_) for ent in doc.ents]
                workflow_result += "\nEntities: " + ", ".join([f"{e[0]}({e[1]})" for e in ents])
            elif step == "TTS" and gTTS:
                tts = gTTS(workflow_result, lang=lang)
                tts_fp = BytesIO()
                tts.write_to_fp(tts_fp)
                tts_fp.seek(0)
                b64 = base64.b64encode(tts_fp.read()).decode()
                st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")
        st.success("Workflow result:")
        st.code(workflow_result, language="text")

# --- Timeline Extraction Tab ---
with tabs[9]:
    st.header("Timeline Extraction & Visualization")
    content = st.session_state["results"]
    if not content or not nlp:
        st.info("No content loaded or spaCy not available.")
    else:
        doc = nlp(content)
        events = []
        for sent in doc.sents:
            if any(ent.label_ == "DATE" for ent in sent.ents):
                date = next((ent.text for ent in sent.ents if ent.label_ == "DATE"), None)
                events.append((date, sent.text))
        if events:
            st.markdown("**Timeline of Events:**")
            for date, event in sorted(events):
                st.markdown(f"- **{date}**: {event}")
        else:
            st.info("No dated events found in the text.")

# --- Topic Modeling Tab ---
with tabs[10]:
    st.header("Topic Modeling & Clustering")
    content = st.session_state["results"]
    if not content:
        st.info("No content loaded. Please use the Analyze tab first.")
    else:
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            import numpy as np
            sentences = [s for s in content.split('.') if len(s.split()) > 3]
            if len(sentences) < 3:
                st.info("Not enough content for topic modeling.")
            else:
                vectorizer = CountVectorizer(stop_words='english')
                X = vectorizer.fit_transform(sentences)
                lda = LatentDirichletAllocation(n_components=3, random_state=42)
                lda.fit(X)
                words = np.array(vectorizer.get_feature_names_out())
                st.markdown("**Main Topics:**")
                for idx, topic in enumerate(lda.components_):
                    top_words = words[np.argsort(topic)[-5:]]
                    st.markdown(f"- Topic {idx+1}: {', '.join(top_words)}")
        except Exception as e:
            st.warning(f"Topic modeling requires scikit-learn and numpy: {e}")

# --- Explainability Tab ---
with tabs[11]:
    st.header("Explainability & Model Transparency")
    content = st.session_state["results"]
    ai_outputs = st.session_state["ai_outputs"]
    if not content or not ai_outputs:
        st.info("No analysis results to explain. Use the AI Tools tab first.")
    else:
        # Show model confidence for sentiment
        if "sentiment" in ai_outputs and "sentiment" in ai_outputs:
            st.markdown("**Sentiment Model Confidence:**")
            sentiment = get_sentiment()
            if sentiment:
                try:
                    sent = sentiment(content, return_all_scores=True)[0]
                    for s in sent:
                        st.write(f"{s['label']}: {s['score']:.2f}")
                except Exception as e:
                    st.warning(f"Could not get confidence scores: {e}")
        # Highlight influential words for NER
        if "ner" in ai_outputs and nlp:
            st.markdown("**Named Entities Highlighted:**")
            doc = nlp(content)
            highlighted = content
            for ent in doc.ents:
                highlighted = highlighted.replace(ent.text, f"**[{ent.text} ({ent.label_})]**")
            st.markdown(highlighted)
        # Show summary with important sentences bolded (rudimentary)
        if "summary" in ai_outputs:
            st.markdown("**Summary with Key Sentences:**")
            summary = ai_outputs["summary"]
            for sent in summary.split('. '):
                if any(word in sent for word in ["important", "key", "notable", "critical"]):
                    st.markdown(f"**{sent}**.")
                else:
                    st.markdown(f"{sent}.")

# --- Settings Tab ---
with tabs[12]:
    st.header("Settings & Privacy")
    st.markdown("- All processing is local unless you use a public API.\n- You can clear all data below.")
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.success("All session data cleared.")

# --- History Tab ---
with tabs[13]:
    st.header("History")
    st.write("View your previous results and chat history.")
    if st.session_state.get("results"):
        st.markdown(f"**Last Analysis Result:**\n\n{st.session_state['results']}")
    if st.session_state.get("ai_outputs"):
        st.markdown("**AI Outputs:**")
        for k, v in st.session_state["ai_outputs"].items():
            st.markdown(f"- **{k.title()}:** {v}")
    if st.session_state.get("chat_history"):
        st.markdown("**Chat History:**")
        for sender, message in st.session_state["chat_history"]:
            st.markdown(f"**{sender}:** {message}")
    if st.session_state.get("batch_results"):
        st.markdown("**Batch Results:**")
        for fname, text in st.session_state["batch_results"]:
            st.markdown(f"**{fname}:**")
            st.code(text, language="text")
    else:
        st.write("No history yet.")

# --- Export Tab ---
with tabs[14]:
    st.header("Export & Sharing")
    content = st.session_state["results"]
    ai_outputs = st.session_state["ai_outputs"]
    if not content:
        st.info("No content to export. Use the Analyze tab first.")
    else:
        st.markdown("**Export your results:**")
        # Export as Markdown
        if st.button("Export as Markdown"):
            md = f"# Analysis Result\n\n{content}\n\n## AI Outputs\n" + "\n".join([f"**{k.title()}**: {v}" for k, v in ai_outputs.items()])
            st.download_button("Download Markdown", md, file_name="obsidian_output.md")
        # Export as PDF (requires fpdf)
        try:
            from fpdf import FPDF
            if st.button("Export as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, content)
                for k, v in ai_outputs.items():
                    pdf.multi_cell(0, 10, f"{k.title()}: {v}")
                pdf_fp = BytesIO()
                pdf.output(pdf_fp)
                pdf_fp.seek(0)
                st.download_button("Download PDF", pdf_fp, file_name="obsidian_output.pdf")
        except Exception as e:
            st.info("Install fpdf for PDF export: pip install fpdf")
        # Export as Word (requires python-docx)
        try:
            from docx import Document
            if st.button("Export as Word"):
                doc = Document()
                doc.add_heading("Analysis Result", 0)
                doc.add_paragraph(content)
                doc.add_heading("AI Outputs", level=1)
                for k, v in ai_outputs.items():
                    doc.add_paragraph(f"{k.title()}: {v}")
                doc_fp = BytesIO()
                doc.save(doc_fp)
                doc_fp.seek(0)
                st.download_button("Download Word", doc_fp, file_name="obsidian_output.docx")
        except Exception as e:
            st.info("Install python-docx for Word export: pip install python-docx")

# --- Accessibility Tab ---
with tabs[15]:
    st.header("Accessibility & Display Settings")
    st.markdown("**Adjust font size and enable dyslexia-friendly font.**")
    font_size = st.slider("Font Size", 12, 32, 16)
    dyslexia_font = st.checkbox("Enable Dyslexia-Friendly Font")
    st.markdown(f"<style>body, .stApp, .stMarkdown, .stTextInput, .stTextArea, .stButton, .stSidebar {{ font-size: {font_size}px !important; {'font-family: OpenDyslexic, Arial, sans-serif !important;' if dyslexia_font else ''} }}</style>", unsafe_allow_html=True)
    st.info("Font size and font will update on next rerun or tab switch.")

# --- User Profiles & Saved Sessions Tab ---
with tabs[16]:
    st.header("User Profiles & Saved Sessions")
    username = st.text_input("Enter your username to save/load your session:")
    if st.button("Save Session") and username:
        st.session_state[f"profile_{username}"] = {
            "results": st.session_state["results"],
            "ai_outputs": st.session_state["ai_outputs"],
            "chat_history": st.session_state["chat_history"]
        }
        st.success(f"Session saved for {username}.")
    if st.button("Load Session") and username and f"profile_{username}" in st.session_state:
        profile = st.session_state[f"profile_{username}"]
        st.session_state["results"] = profile["results"]
        st.session_state["ai_outputs"] = profile["ai_outputs"]
        st.session_state["chat_history"] = profile["chat_history"]
        st.success(f"Session loaded for {username}.")
    st.info("Sessions are saved locally in your browser session.")

# --- Real-Time Collaboration Placeholder Tab ---
with tabs[17]:
    st.header("Real-Time Collaboration (Coming Soon)")
    st.info("This feature will allow multiple users to analyze and discuss the same document in real time.")
    st.markdown("If you want to help build this, open an issue or PR!")

# --- Data Visualization Tab ---
with tabs[18]:
    st.header("Data Visualizations")
    import matplotlib.pyplot as plt
    import collections
    content = st.session_state["results"]
    ai_outputs = st.session_state["ai_outputs"]
    if not content or not ai_outputs:
        st.info("No analysis results to visualize. Use the AI Tools tab first.")
    else:
        # Entity frequency
        if "ner" in ai_outputs and ai_outputs["ner"]:
            st.subheader("Entity Frequency")
            ents = [ent for ent, label in ai_outputs["ner"]]
            counter = collections.Counter(ents)
            fig, ax = plt.subplots()
            ax.bar(counter.keys(), counter.values(), color="#001F3F")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        # Sentiment pie chart
        if "sentiment" in ai_outputs and "POSITIVE" in ai_outputs["sentiment"] or "NEGATIVE" in ai_outputs["sentiment"]:
            st.subheader("Sentiment Distribution")
            labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            values = [ai_outputs["sentiment"].count(l) for l in labels]
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=["#0074D9", "#FF4136", "#AAAAAA"])
            st.pyplot(fig)
        # Topic bar chart
        if "topics" in ai_outputs and ai_outputs["topics"]:
            st.subheader("Topic Prevalence")
            topics = ai_outputs["topics"]
            fig, ax = plt.subplots()
            ax.bar([f"Topic {i+1}" for i in range(len(topics))], [len(t) for t in topics], color="#39CCCC")
            st.pyplot(fig)

# --- Theme Picker Tab ---
with tabs[19]:
    st.header("Customizable Themes")
    theme = st.selectbox("Choose a theme:", ["Black & Navy Blue", "Classic Dark", "Light", "Solarized"])
    if theme == "Black & Navy Blue":
        st.markdown('<style>body, .stApp { background-color: #000014; color: #E4E6EB; } .stTabs [aria-selected="true"] { background: #001F3F; color: #FFD700; }</style>', unsafe_allow_html=True)
    elif theme == "Classic Dark":
        st.markdown('<style>body, .stApp { background-color: #18191A; color: #E4E6EB; } .stTabs [aria-selected="true"] { background: #3A3B3C; color: #FFD700; }</style>', unsafe_allow_html=True)
    elif theme == "Light":
        st.markdown('<style>body, .stApp { background-color: #FFFFFF; color: #222; } .stTabs [aria-selected="true"] { background: #E0E0E0; color: #0074D9; }</style>', unsafe_allow_html=True)
    elif theme == "Solarized":
        st.markdown('<style>body, .stApp { background-color: #002B36; color: #839496; } .stTabs [aria-selected="true"] { background: #073642; color: #B58900; }</style>', unsafe_allow_html=True)
    st.info("Theme will update on next rerun or tab switch.")

# Footer
st.markdown("<hr style='margin-top:2em;margin-bottom:1em'>", unsafe_allow_html=True)
st.caption("Obsidian Protocol v2.0 | World-Changing Streamlit UI | All media types & AI tools | Free forever | by AI")
