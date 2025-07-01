Absolutely! Here is your final, double-checked, ready-to-copy README for GitHub:

---

# 🧠 Obsidian Protocol

> A system that reveals the truth inside media, speeches, and noise — using AI.

---

## 🌐 Purpose

The world is full of manipulated messages. Obsidian analyzes raw text (articles, speeches, social posts, media files) and gives you:

- 📄 A neutral **summary**
- ⚖️ A breakdown of **bias, emotion, or manipulation**
- 🕵️‍♂️ A list of **key names** (people, orgs, countries) mentioned
- 🗣️ Audio, image, and video analysis
- 🌍 Multilingual support and translation
- 🔊 Text-to-speech and speech-to-text
- 🧠 Q&A, knowledge graphs, and more

No spin. No filters. Just signal.

---

## 🚀 Features

- **Text, TXT, PDF, URL, YouTube, Image, Audio** input (single & batch)
- **Summarization** (HuggingFace Transformers)
- **Sentiment Analysis** (HuggingFace Transformers)
- **Named Entity Recognition (NER)** (spaCy)
- **Translation** (multi-language, HuggingFace)
- **Text-to-Speech (TTS)** (gTTS, pyttsx3)
- **Speech-to-Text** (Whisper)
- **Document Q&A** (local QA pipeline)
- **Image Captioning** (BLIP model)
- **Image OCR** (pytesseract)
- **Audio Sentiment & Speaker Diarization** (pyAudioAnalysis)
- **Batch/Multiple File Processing**
- **Knowledge Graph Extraction** (networkx + pyvis)
- **Plagiarism/Similarity Detection** (sentence-transformers, cosine similarity)
- **Customizable Workflows** (chain tools in any order)
- **Download Results** (as .txt, audio, etc.)
- **Privacy Controls** (clear all session data)
- **History** (view all previous results, chat, batch outputs)
- **Modular, beautiful, dark-mode UI** (Streamlit, custom CSS)
- **Multilingual support** (for translation, TTS, and more)
- **No OpenAI API required** (all local or free/public models/APIs)
- **Free forever** (no usage limits, no paywall)

---

## 🛠️ How It Works

- Built with:  
  `Streamlit`, `HuggingFace Transformers`, `SpaCy NLP`, `PyPDF2`, `Pillow`, `pytesseract`, `openai-whisper`, `networkx`, `pyvis`, `sentence-transformers`, `pyAudioAnalysis`, `gTTS`, `pyttsx3`
- Input: Any article, speech, media file, or batch
- Output:
  - AI-generated **summary**
  - **Sentiment analysis**
  - Top **named entities** detected
  - Translations, TTS, Q&A, captions, graphs, and more

---

## 🧪 Example Input

> “In a statement, Elon Musk warned AI could be humanity’s greatest risk, while OpenAI and Google continue to expand their systems.”

### 🧠 Output

```json
{
  "summary": "Elon Musk warns about AI as OpenAI and Google expand.",
  "tone": {
    "label": "NEGATIVE",
    "score": 0.93
  },
  "key_entities": [
    ["Elon Musk", 1],
    ["OpenAI", 1],
    ["Google", 1]
  ]
}
```

---

