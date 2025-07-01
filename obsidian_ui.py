import streamlit as st
import tempfile
import os

# Try to import whisper, else show install instructions
try:
    import whisper
    whisper_available = True
except ImportError:
    whisper_available = False

st.set_page_config(page_title="Obsidian Protocol v2.0", layout="wide")

# Sidebar
st.sidebar.title("🧠 Obsidian Protocol v2.0")
st.sidebar.markdown("AI-powered Media Analyzer")
st.sidebar.info("Upload your media and get instant AI insights.")

# Tabs
TABS = ["Analyze File", "Live Chat", "History"]
tab1, tab2, tab3 = st.tabs(TABS)

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "transcription" not in st.session_state:
    st.session_state["transcription"] = None

with tab1:
    st.header("Analyze File")
    uploaded_file = st.file_uploader(
        "Upload a media file (audio, video, image, or document)",
        type=["mp3", "wav", "mp4", "mov", "jpg", "jpeg", "png", "pdf", "txt"],
        help="Supported formats: audio, video, image, PDF, text."
    )
    user_prompt = st.text_area(
        "Enter notes or an AI prompt (optional)",
        placeholder="E.g., Summarize this file, extract key points, etc."
    )
    analyze = st.button("Analyze with AI", key="analyze_btn")

    if uploaded_file is not None:
        st.info(f"Uploaded: {uploaded_file.name}")
        # If audio file and whisper is available, transcribe
        if uploaded_file.type.startswith("audio") and whisper_available:
            st.write("Transcribing audio with Whisper...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            st.session_state["transcription"] = result["text"]
            st.success("Transcription complete!")
            st.markdown(f"**Transcription:**\n\n{st.session_state['transcription']}")
            os.remove(tmp_path)
        elif uploaded_file.type.startswith("audio") and not whisper_available:
            st.warning("Whisper is not installed. To enable audio transcription, run: pip install openai-whisper")
        # Placeholder for other file types
        else:
            st.info("File preview and analysis coming soon.")
    if analyze:
        st.success("AI analysis results will appear here. (Integration pending)")

    st.markdown("---")
    st.subheader("AI Analysis Results")
    st.write("Results will be displayed here after analysis.")
    if st.session_state["transcription"]:
        st.markdown(f"**Last Transcription:**\n\n{st.session_state['transcription']}")

with tab2:
    st.header("Live Chat with AI")
    st.write("Chat with the AI about your uploaded files or transcriptions.")
    chat_input = st.text_input("You:", key="chat_input")
    if st.button("Send", key="send_btn") and chat_input:
        # Placeholder for AI response
        ai_response = f"[AI]: (This is a placeholder response to: '{chat_input}')"
        st.session_state["chat_history"].append(("You", chat_input))
        st.session_state["chat_history"].append(("AI", ai_response))
    # Display chat history
    for sender, message in st.session_state["chat_history"][-10:]:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"<span style='color: #4F8BF9'><b>AI:</b> {message}</span>", unsafe_allow_html=True)
    if st.session_state["transcription"]:
        st.info(f"You can reference the last transcription in your chat.")

with tab3:
    st.header("History")
    st.write("View your previous transcriptions and chat history.")
    if st.session_state["transcription"]:
        st.markdown(f"**Transcription History:**\n\n{st.session_state['transcription']}")
    if st.session_state["chat_history"]:
        st.markdown("**Chat History:**")
        for sender, message in st.session_state["chat_history"]:
            st.markdown(f"**{sender}:** {message}")
    else:
        st.write("No history yet.")

# Footer
st.markdown("<hr style='margin-top:2em;margin-bottom:1em'>", unsafe_allow_html=True)
st.caption("Obsidian Protocol v2.0 | Streamlit UI | Whisper audio transcription enabled | Live chat placeholder")
