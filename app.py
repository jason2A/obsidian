from flask import Flask, request, jsonify
from transformers import pipeline
import spacy
from collections import Counter

app = Flask(__name__)

# Load NLP tools
summarizer = pipeline("summarization")
sentiment = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")  # Named Entity Recognition (NER)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    # 1. Summarize
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # 2. Sentiment / Bias Tone
    tone = sentiment(summary)[0]

    # 3. Extract Entities
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    top_entities = Counter(entities).most_common(5)

    return jsonify({
        "summary": summary,
        "tone": tone,
        "key_entities": top_entities
    })

if __name__ == '__main__':
    app.run(debug=True)
