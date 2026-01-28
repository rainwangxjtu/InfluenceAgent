# tts.py
import asyncio
import os
import edge_tts

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

async def _save_mp3(text: str, out_path: str, voice: str):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_path)

def tts_zh(text: str, out_path: str, voice: str = DEFAULT_VOICE):
    if not text or not text.strip():
        raise ValueError("TTS received empty text")

    tmp_path = out_path + ".tmp"

    # Remove any old temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Generate to temp first
    asyncio.run(_save_mp3(text, tmp_path, voice))

    # Validate file actually has content
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError("TTS failed: generated file is empty (0 bytes). Check network / edge-tts.")

    # Atomically replace output
    os.replace(tmp_path, out_path)
