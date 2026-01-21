# streamlit_app.py
import streamlit as st
from app import InfluenceAgent, process_text_input, process_audio_input

st.set_page_config(page_title="InfluenceAgent", layout="centered")

st.title("🎧 InfluenceAgent")
st.subheader("Cross-Lingual Podcast-to-Chinese AI System")

# Initialize the agent once
@st.cache_resource
def load_agent():
    return InfluenceAgent()

agent = load_agent()

# Input mode selection
mode = st.radio("Choose input mode:", ["Text", "Audio"])
audience = st.selectbox("Target Audience:", ["general", "student", "professional"])

# -----------------------
# Text Input Mode
# -----------------------
if mode == "Text":
    text_input = st.text_area("Enter English podcast text:", height=200)

    if st.button("Generate Chinese Output"):
        if not text_input.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Processing..."):
                output = process_text_input(agent, text_input, audience)
                st.success("Done!")
                st.markdown("### Chinese Output")
                st.write(output)

# -----------------------
# Audio Input Mode
# -----------------------
else:
    audio_file = st.file_uploader(
        "Upload an English podcast audio file", type=["wav", "mp3", "m4a"]
    )

    if audio_file and st.button("Generate Chinese Output"):
        with st.spinner("Processing..."):
            output = process_audio_input(agent, audio_file, audience)
            st.success("Done!")
            st.markdown("### Chinese Output")
            st.write(output)

# -----------------------
# Notes
# -----------------------
st.markdown(
    """
    **Notes:**  
    - Audio processing uses OpenAI Whisper (`base` model).  
    - Summarization uses BART; Translation uses MarianMT; Audience adaptation uses FLAN-T5.  
    - Logs are written to `influenceagent.log`.  
    """
)
