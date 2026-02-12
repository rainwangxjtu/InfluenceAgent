# app.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import subprocess
from pathlib import Path

import whisper
from transformers import pipeline

from logger import log_event
from evaluation import summary_stats

import torch

def pick_hf_device():
    if torch.cuda.is_available():
        return 0                      # CUDA GPU index
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")    # Apple Silicon / MPS
    return -1                         # CPU

class InfluenceAgent:
    def __init__(self):
        print("Loading Whisper ASR model...")
        self.asr = whisper.load_model("base")

        print("Loading NLP models...")
        hf_device = pick_hf_device()

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=hf_device)
        self.translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=hf_device)
        self.generator  = pipeline("text2text-generation", model="google/flan-t5-small", device=hf_device)

        import torch
        gen_device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("text2text-generation", model="google/flan-t5-small", device=gen_device)

    def save_text(self, path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")

    def transcribe_audio(self, audio_path: str) -> str:
        log_event("ASR", f"Transcribing file: {audio_path}")
        t0 = time.time()
        result = self.asr.transcribe(audio_path)
        text = (result.get("text") or "").strip()
        log_event("ASR", f"Completed in {time.time() - t0:.2f}s")
        return text

    def summarize(self, text: str, mode="short") -> str:
        log_event("Summarization", f"Mode={mode}, Input length={len(text)}")
        t0 = time.time()

        text = (text or "")[:5000]
        input_words = max(1, len(text.split()))

        if mode == "short":
            max_len = min(80, max(40, int(input_words * 0.45)))
            min_len = min(25, max(10, int(max_len * 0.5)))
        else:
            max_len = min(180, max(80, int   (input_words * 0.75)))
            min_len = min(60, max(30, int(max_len * 0.5)))

    # guard against invalid constraints
        max_len = max(max_len, 10)
        min_len = max(5, min(min_len, max_len - 1))

        out = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )[0]["summary_text"]

        log_event("Summarization", f"Completed in {time.time() - t0:.2f}s")
        return (out or "").strip()

    def translate(self, text: str) -> str:
        log_event("Translation", f"Input length={len(text)}")
        t0 = time.time()
        zh_text = self.translator(text)[0]["translation_text"]
        log_event("Translation", f"Completed in {time.time() - t0:.2f}s")
        return (zh_text or "").strip()

    def adapt_for_audience(self, text: str, audience="general") -> str:
        log_event("Adaptation", f"Audience={audience}, Input length={len(text)}")
        t0 = time.time()

        prompt = (
            f"你是一名中文内容创作助手。请将下面内容改写成面向{audience}受众的中文口播讲稿。\n"
            "要求：\n"
            "1) 只输出中文\n"
            "2) 不要输出空白或无意义字符\n"
            "3) 结构包含：开头点题；3条要点；结尾总结\n"
            "4) 字数不少于200字\n\n"
            f"内容：\n{text}\n\n中文口播讲稿："
        )

        out = self.generator(prompt, max_new_tokens=320, do_sample=False, num_beams=4)[0]["generated_text"]
        adapted = (out or "").strip()

        if len(adapted) < 80:
            log_event("Adaptation", f"Retry: output too short (len={len(adapted)})")
            prompt2 = (
                "请用中文把下面内容改写成一段适合学生听的口播讲稿，"
                "至少200字，必须包含3条要点，不能留空。\n\n"
                f"{text}\n\n中文讲稿："
            )
            out2 = self.generator(prompt2, max_new_tokens=360, do_sample=False, num_beams=4)[0]["generated_text"]
            adapted = (out2 or "").strip()

        log_event("Adaptation", f"Completed in {time.time() - t0:.2f}s; output_len={len(adapted)}")
        return adapted

    def tts_zh(self, zh_text: str, out_path: str, voice: str = "Tingting", chunk_chars: int = 250):
        """
        macOS TTS using `say`. Chunks long text to avoid failures.
        Writes AIFF then converts to WAV if ffmpeg exists and out_path ends with .wav
        """
        zh_text = (zh_text or "").strip()
        if not zh_text:
            log_event("TTS", "Empty zh_text; skipping TTS.")
            return None

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        aiff_path = out_path.with_suffix(".aiff")

        def chunk_text(s: str, n: int) -> list[str]:
            import re
            parts = re.split(r"(?<=[。！？；\n])", s)
            parts = [p.strip() for p in parts if p and p.strip()]
            chunks, buf = [], ""
            for p in parts:
                if not buf:
                    buf = p
                elif len(buf) + len(p) + 1 <= n:
                    buf += " " + p
                else:
                    chunks.append(buf)
                    buf = p
            if buf:
                chunks.append(buf)
            return chunks

        chunks = chunk_text(zh_text, chunk_chars)

        tmp_dir = out_path.parent / "_tts_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        piece_paths = []

        try:
            for i, ch in enumerate(chunks):
                piece = tmp_dir / f"piece_{i:03d}.aiff"
                subprocess.run(["say", "-v", voice, "-o", str(piece), ch], check=True)
                piece_paths.append(piece)

            # concat pieces -> one AIFF (requires ffmpeg if >1 piece)
            if len(piece_paths) == 1:
                piece_paths[0].replace(aiff_path)
            else:
                try:
                    concat_list = tmp_dir / "concat.txt"
                    concat_list.write_text(
                        "\n".join([f"file '{p.as_posix()}'" for p in piece_paths]),
                        encoding="utf-8",
                    )
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(aiff_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    # fallback: try one-shot say (may fail on long input, but last resort)
                    log_event("TTS", "ffmpeg not found; falling back to one-shot `say` (may fail on long text).")
                    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), zh_text], check=True)

            # convert to WAV if requested
            if out_path.suffix.lower() == ".wav":
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(aiff_path), str(out_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    log_event("TTS", f"Wrote WAV: {out_path}")
                    return str(out_path)
                except FileNotFoundError:
                    log_event("TTS", f"ffmpeg not found. Keeping AIFF: {aiff_path}")
                    return str(aiff_path)

            log_event("TTS", f"Wrote AIFF: {aiff_path}")
            return str(aiff_path)

        finally:
            for p in piece_paths:
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                (tmp_dir / "concat.txt").unlink()
            except Exception:
                pass
            try:
                tmp_dir.rmdir()
            except Exception:
                pass

    def run_from_audio(self, audio_path: str, audience="general", out_dir: str | Path = "outputs/pilot"):
        out_dir = Path("outputs/pilot")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) ASR
        text = self.transcribe_audio(audio_path)
        self.save_text(out_dir / "asr_en.txt", text)

        # 2) Summaries
        short = self.summarize(text, "short")
        long = self.summarize(text, "long")
        self.save_text(out_dir / "sum_short_en.txt", short)
        self.save_text(out_dir / "sum_long_en.txt", long)

        # 3) Translate
        zh_short = self.translate(short)
        zh_long = self.translate(long)
        self.save_text(out_dir / "sum_short_zh.txt", zh_short)
        self.save_text(out_dir / "sum_long_zh.txt", zh_long)

        # 4) Rewrite/adapt
        adapted = self.adapt_for_audience(zh_long, audience)
        if not adapted.strip():
            log_event("Adaptation", "Empty adaptation output. Falling back to zh_long.")
            adapted = zh_long
        self.save_text(out_dir / "rewrite_zh.txt", adapted)

        # 5) TTS
        tts_file = self.tts_zh(adapted, out_path=str(out_dir / "tts_zh.wav"), voice="Tingting")

        # Eval
        stats = summary_stats(text, short)
        log_event("Evaluation", str(stats))

        return adapted, tts_file


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    ap = argparse.ArgumentParser(
        prog="InfluenceAgent",
        description="English audio -> ASR -> summarize -> translate -> adapt -> Chinese TTS (wav/aiff).",
    )
    ap.add_argument(
        "--audio",
        required=True,
        help="Path to input audio (mp3/wav/m4a).",
    )
    ap.add_argument(
        "--audience",
        default="student",
        help='Audience style for Chinese adaptation (e.g., "student", "intermediate learners").',
    )
    ap.add_argument(
        "--out",
        default="outputs/pilot",
        help="Output directory for text + audio artifacts.",
    )
    ap.add_argument(
        "--voice",
        default="Tingting",
        help='macOS `say` voice name (e.g., Tingting, Mei-Jia).',
    )
    ap.add_argument(
        "--no_tts",
        action="store_true",
        help="Skip TTS and only write text outputs.",
    )
    ap.add_argument(
        "--tts_name",
        default="tts_zh.wav",
        help="TTS output filename (wav or aiff).",
    )

    args = ap.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out)

    if not audio_path.exists():
        raise SystemExit(f"Missing audio file: {audio_path}")

    # Optional: reduce tokenizers fork warning noise
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    agent = InfluenceAgent()

    # Let run_from_audio use your chosen output dir (recommended small refactor below)
    # For now, simplest: set out_dir inside agent or pass it into run_from_audio.
    adapted, tts_file = agent.run_from_audio(str(audio_path), audience=args.audience, out_dir=out_dir)

    # If you want --no_tts / --voice / --tts_name to work, do TTS here:
    if args.no_tts:
        print("✅ Done. (TTS skipped)")
        raise SystemExit(0)

    tts_out = out_dir / args.tts_name
    tts_file = agent.tts_zh(adapted, out_path=str(tts_out), voice=args.voice)

    print("✅ Done.")
    print(f"- audio_in: {audio_path}")
    print(f"- out_dir : {out_dir}")
    print(f"- tts     : {tts_file}")
