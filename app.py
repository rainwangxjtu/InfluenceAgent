# app.py
import whisper
from transformers import pipeline
import os
import time
from logger import log_event
from evaluation import summary_stats

class InfluenceAgent:
    def __init__(self):
        print("Loading Whisper ASR model...")
        self.asr = whisper.load_model("base")

        print("Loading NLP models...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

    def transcribe_audio(self, audio_path):
        log_event("ASR", f"Transcribing file: {audio_path}")
        start = time.time()

        result = self.asr.transcribe(audio_path)
        text = result["text"]

        end = time.time()
        log_event("ASR", f"Completed in {end - start:.2f}s")

        return text

    def summarize(self, text, mode="short"):
        log_event("Summarization", f"Mode={mode}, Input length={len(text)}")
        start = time.time()

        if mode == "short":
            max_len = 60
        else:
            max_len = 250

        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]

        end = time.time()
        log_event("Summarization", f"Completed in {end - start:.2f}s")

        return summary

    def translate(self, text):
        log_event("Translation", f"Input length={len(text)}")
        start = time.time()

        zh_text = self.translator(text)[0]["translation_text"]

        end = time.time()
        log_event("Translation", f"Completed in {end - start:.2f}s")

        return zh_text

    def adapt_for_audience(self, text, audience="general"):
        log_event("Adaptation", f"Audience={audience}, Input length={len(text)}")
        start = time.time()

        prompt = f"Rewrite the following Chinese text for a {audience} audience:\n{text}"

        adapted = self.generator(
            prompt,
            max_length=300,
            min_length=50,
            do_sample=False
        )[0]["generated_text"]

        end = time.time()
        log_event("Adaptation", f"Completed in {end - start:.2f}s")

        return adapted

    def run_from_audio(self, audio_path, audience="general"):
        text = self.transcribe_audio(audio_path)
        return self.run_from_text(text, audience)

    def run_from_text(self, text, audience="general"):
        print("\n--- Short Summary (EN) ---")
        short = self.summarize(text, "short")
        print(short)

        print("\n--- Long Summary (EN) ---")
        long = self.summarize(text, "long")
        print(long)

        print("\n--- Short Summary (ZH) ---")
        zh_short = self.translate(short)
        print(zh_short)

        print("\n--- Long Summary (ZH) ---")
        zh_long = self.translate(long)
        print(zh_long)

        print("\n--- Student-Oriented Chinese Output ---")
        adapted = self.adapt_for_audience(zh_long, audience)
        print(adapted)

        stats = summary_stats(text, short)
        log_event("Evaluation", str(stats))

        return adapted


if __name__ == "__main__":
    agent = InfluenceAgent()

    AUDIO_FILE = "sample_audio.mp3"

    if os.path.exists(AUDIO_FILE):
        agent.run_from_audio(AUDIO_FILE, audience="student")
    else:
        with open("sample_input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        agent.run_from_text(text, audience="student")
