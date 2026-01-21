#python3

import streamlit as st
from app import process_text_input, process_audio_input

st.set_page_config(page_title="InfluenceAgent", layout="centered")

st.title("🎧 InfluenceAgent")
st.subheader("Cross-Lingual Podcast-to-Chinese AI System")

mode = st.radio("Choose input mode:", ["Text", "Audio"])
audience = st.selectbox("Target Audience:", ["general", "student", "professional"])

if mode == "Text":
    text_input = st.text_area("Enter English podcast text:", height=200)
    if st.button("Generate Chinese Output"):
        with st.spinner("Processing..."):
            result = process_text_input(text_input, audience)
            st.success("Done!")
            st.markdown("### Chinese Output")
            st.write(result)

else:
    audio_file = st.file_uploader("Upload an English podcast audio file", type=["wav", "mp3", "m4a"])
    if audio_file and st.button("Generate Chinese Output"):
        with st.spinner("Processing..."):
            result = process_audio_input(audio_file, audience)
            st.success("Done!")
            st.markdown("### Chinese Output")
            st.write(result)
