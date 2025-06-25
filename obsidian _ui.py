import streamlit as st
from transformers import pipeline
import spacy

# Load NLP tools
summarizer = pipeline("summarization")
classifier = pipeline("sentiment-analysis")
import spacy.cli
spacy.cli.download("en_core_web_sm")
ner = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Obsidian Protocol", layout="wide")

st.title("🧠 Obsidian Protocol")
st.subheader("Reveal the truth behind any media, speech, or post.")

user_input = st.text_area("Paste your article, speech, or social post here:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some text to analyze.")
    else:
        # Generate summary
        summary = summarizer(user_input, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

        # Get sentiment
        tone = classifier(user_input)[0]

        # Extract entities
        doc = ner(user_input)
        entities = list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]))

        st.success("✅ Analysis Complete")
        
        st.markdown("### 📝 Summary")
        st.write(summary)

        st.markdown("### 🎭 Tone")
        st.write(f"Label: `{tone['label']}` | Score: `{round(tone['score'], 2)}`")

        st.markdown("### 🕵️ Key Entities")
        if entities:
            st.write(", ".join(entities))
        else:
            st.write("No major people, organizations, or locations found.")
