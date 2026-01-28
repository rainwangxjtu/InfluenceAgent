# audio2audio_test.py
from app import InfluenceAgent
from tts import tts_zh
import os

def main():
    AUDIO_IN = "sample_audio.mp3"
    AUDIO_OUT = "zh_output.mp3"
    TEXT_OUT = "zh_output.txt"
    AUDIENCE = "student"

    if not os.path.exists(AUDIO_IN):
        raise FileNotFoundError(f"{AUDIO_IN} not found")

    print("🔊 Loading InfluenceAgent...", flush=True)
    agent = InfluenceAgent()

    print("➡️ Starting English audio → Chinese text...", flush=True)
    zh_text = agent.run_from_audio(AUDIO_IN, audience=AUDIENCE)
    zh_text = zh_text.strip()
    print("✅ Finished audio → Chinese text", flush=True)

    # Save Chinese text for inspection
    with open(TEXT_OUT, "w", encoding="utf-8") as f:
        f.write(zh_text)

    print("\n--- zh_text preview ---", flush=True)
    print(zh_text[:300], flush=True)
    print(f"\nzh_text length: {len(zh_text)} characters", flush=True)
    print(f"✅ Saved Chinese text to: {TEXT_OUT}", flush=True)

    print("\n➡️ Starting Chinese text → Chinese audio (TTS)...", flush=True)
    tts_zh(zh_text, AUDIO_OUT)

    # Verify MP3 was written
    size = os.path.getsize(AUDIO_OUT)
    print(f"✅ MP3 written: {AUDIO_OUT} ({size} bytes)", flush=True)

if __name__ == "__main__":
    main()
