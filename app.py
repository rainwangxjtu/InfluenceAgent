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
        self.generator = pipeline("text2text-generation",model="google/flan-t5-small",device=0)

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

    # Keep input manageable for speed
        text = text[:5000]

    # Estimate input size (rough)
        input_words = max(1, len(text.split()))

        if mode == "short":
            max_len = min(80, max(40, int(input_words * 0.45)))
            min_len = min(25, max(10, int(max_len * 0.5)))
        else:
            max_len = min(180, max(80, int(input_words * 0.75)))
            min_len = min(60, max(30, int(max_len * 0.5)))

        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]["summary_text"]

        log_event("Summarization", f"Completed in {time.time() - start:.2f}s")
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

    # Strong Chinese-only instruction to avoid empty outputs
        prompt = (
            f"你是一名中文内容创作助手。请将下面内容改写成面向{audience}受众的中文口播讲稿。\n"
            "要求：\n"
            "1) 只输出中文\n"
            "2) 不要输出空白或无意义字符\n"
            "3) 结构包含：开头点题；3条要点；结尾总结\n"
            "4) 字数不少于200字\n\n"
            f"内容：\n{text}\n\n中文口播讲稿："
        )

    # Try once
        out = self.generator(
            prompt,
            max_new_tokens=320,
            do_sample=False,
            num_beams=4
        )[0]["generated_text"]

        adapted = (out or "").strip()

    # Retry if empty/too short (common on some setups)
        if len(adapted) < 80:
            log_event("Adaptation", f"Retry: output too short (len={len(adapted)}). Falling back to safer prompt.")
            prompt2 = (
                "请用中文把下面内容改写成一段适合学生听的口播讲稿，"
                "至少200字，必须包含3条要点，不能留空。\n\n"
                f"{text}\n\n中文讲稿："
            )
            out2 = self.generator(
                prompt2,
                max_new_tokens=360,
                do_sample=False,
                num_beams=4
            )[0]["generated_text"]
            adapted = (out2 or "").strip()

        log_event("Adaptation", f"Completed in {time.time() - start:.2f}s; output_len={len(adapted)}")
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

        if not adapted.strip():
            log_event("Adaptation", "Empty adaptation output. Falling back to zh_long.")
            adapted = zh_long

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