# python3

from transformers import pipeline

class Podcast2ChineseSystem:
    def __init__(self):
        print("Loading models...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

    def summarize(self, text, mode="short"):
        if mode == "short":
            max_len, min_len = 60, 30
        else:
            max_len, min_len = 180, 80

        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]["summary_text"]
        return summary

    def translate(self, text):
        return self.translator(text)[0]["translation_text"]

    def generate_for_audience(self, text, audience="students"):
        prompt = f"Rewrite the following content for {audience} in Chinese, clearly and engagingly:\n{text}"
        return self.generator(prompt, max_length=256)[0]["generated_text"]

    def run(self, text):
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
        student = self.generate_for_audience(zh_long, "Chinese college students")
        print(student)


if __name__ == "__main__":
    with open("sample_input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    system = Podcast2ChineseSystem()
    system.run(text)