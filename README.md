# InfluenceAgent: A Cross-Lingual Podcast-to-Chinese NLP System

InfluenceAgent is a modular NLP pipeline that converts **English podcast audio/text** into **Chinese summaries and audience-adapted scripts**, and can optionally generate **Chinese narration audio**.

This repo is designed as an applied ML/NLP systems demo, emphasizing pipeline design, model integration, reproducibility, and engineering trade-offs.

---

## Overview

InfluenceAgent integrates:
- **Speech Recognition (ASR)**: Whisper (audio → English transcript)
- **Summarization**: Transformer summarization (short/long)
- **Machine Translation**: EN → ZH translation
- **Audience Adaptation**: Controlled generation for different audiences
- **Optional Chinese TTS**: Chinese text → Chinese audio

Supported outputs:
- **Short-form** Chinese content (≤ 1 minute)
- **Long-form** Chinese scripts (10–15 minutes; best with chunking / future work)
- **Chinese audio-ready scripts** (and optional narration audio)

---

## Motivation

High-quality English podcasts contain rich knowledge and timely reporting, but language barriers prevent many Chinese-speaking audiences from accessing them efficiently.

InfluenceAgent addresses this by:
- Automating **cross-lingual knowledge transfer**
- Supporting **audience-aware adaptation** (general / student / professional)
- Offering **length and format control** (short vs. long summaries)

Beyond translation, this project emphasizes:
- Modular system architecture
- Scalability/extensibility
- Engineering trade-offs (latency vs. quality)
- Lightweight evaluation and logging

---

## System Architecture

Pipeline design:

1. **English Podcast (Audio/Text)**
2. **ASR (Whisper)**
3. **Summarization (Short / Long)**
4. **Translation (EN → ZH)**
5. **Audience-Aware Generation**
6. **Chinese Output (Text / Optional Audio)**

Each component is independently replaceable, enabling experimentation with different models, speeds, and quality/latency trade-offs.

---

## Core Components

### 1) Speech Recognition (ASR)
- Model: **OpenAI Whisper**
- Converts English podcast audio into text for downstream processing

### 2) Summarization
- Default model: **BART (facebook/bart-large-cnn)**  
- Produces short/long summaries  
- Note: for short inputs, summarizers may behave extractively; instruction-guided summarization is a possible extension.

### 3) Machine Translation
- Model: **MarianMT (Helsinki-NLP)**
- Translates English summaries into Chinese with emphasis on fidelity + fluency

### 4) Audience-Aware Text Generation
- Model: **FLAN-T5**
- Rewrites Chinese content for:
  - general public
  - students
  - professionals

### 5) Output Layer
- Chinese text for short/long scripts
- Optional **Chinese narration audio** (TTS)

---


Also update your “Repository Structure” to reflect reality (remove files you aren’t shipping / aren’t used anymore), e.g.:

```md
├── app.py           # CLI pipeline (ASR -> summarize -> translate -> adapt -> optional TTS)
├── logger.py        # Logging utilities
├── evaluation.py    # Lightweight evaluation metrics
├── requirements.txt
└── outputs/         # (gitignored) generated artifacts

```
---

## Dependencies

### Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```
---

### System dependency (required for Whisper)
```bash
Whisper requires ffmpeg for audio decoding.

- macOS: brew install ffmpeg

- Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg

Notes: edge-tts requires an internet connection to generate speech.

```
---

## Quickstart: English audio → Chinese text + Chinese audio (TTS)

### 1) Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2) Install ffmpeg (required by Whisper)
macOS: brew install ffmpeg
Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg

3. Run the Pipeline:
python app.py --audio data/clips/sample_audio.mp3 --audience student

outputs/pilot/
  asr_en.txt
  sum_short_en.txt
  sum_long_en.txt
  sum_short_zh.txt
  sum_long_zh.txt
  rewrite_zh.txt
  tts_zh.wav   (or .aiff if ffmpeg isn't available)

macOS playback:
afplay outputs/pilot/tts_zh.wav 2>/dev/null || open outputs/pilot/tts_zh.aiff

Watch logs:
tail -f influenceagent.log

```
---
