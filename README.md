# 🧠 Obsidian Protocol

> A system that reveals the truth inside media, speeches, and noise — using AI.

---

### 🌐 Purpose

The world is full of manipulated messages. Obsidian analyzes raw text (articles, speeches, social posts) and gives you:

- 📄 A neutral **summary**
- ⚖️ A breakdown of **bias, emotion, or manipulation**
- 🕵️‍♂️ A list of **key names** (people, orgs, countries) mentioned

No spin. No filters. Just signal.

---

### 🔧 How It Works

- Built with: `Flask`, `HuggingFace Transformers`, `SpaCy NLP`
- Input: Any long article or speech
- Output:
  - AI-generated **summary**
  - **Sentiment analysis**
  - Top **named entities** detected

---

### 🧪 Example Input:

> “In a statement, Elon Musk warned AI could be humanity’s greatest risk, while OpenAI and Google continue to expand their systems.”

### 🧠 Output:

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
