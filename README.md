# InfluenceAgent: A Cross-Lingual Podcast-to-Chinese NLP System
This project implements a scalable, modular NLP system that converts English podcast content into audience-adapted Chinese multimedia outputs.

🔍 Overview

InfluenceAgent is a scalable, modular Natural Language Processing (NLP) system that converts English podcast content into audience-adapted Chinese multimedia outputs.

By integrating speech recognition (Whisper), summarization, neural machine translation, and controlled text generation, the system enables flexible generation of:

- Short-form Chinese video content (≤ 1 minute)
- Long-form Chinese video/podcast content (10–15 minutes)
- Chinese audio-ready scripts tailored for different audiences

🎯 Motivation

High-quality English podcasts contain rich educational and informational content, but language barriers prevent non-English-speaking audiences from accessing this knowledge efficiently.

InfluenceAgent addresses this gap by:

- Enabling automated cross-lingual knowledge transfer
- Supporting audience-aware content adaptation
- Providing flexible content length and format control

Beyond translation, this project emphasizes:
- Modular system design
- Scalability
- Engineering trade-offs
- Research-driven evaluation

🧠 System Architecture

The system follows a pipeline-based design:

    Step 1: English Podcast (Audio/Text)
                
    Step 2: Speech Recognition (Whisper)
                
    Step 3: Content Summarization (Short / Long)
                
    Step 4: Neural Machine Translation (EN → ZH)
                
    Step 5: Audience-Aware Content Generation
                
    Step 6: Chinese Multimedia Output

Each component is independently replaceable and extensible, allowing experimentation with different models, performance optimizations, and deployment strategies.

⚙️ Core Components
1. Speech Recognition (ASR)
- Model: OpenAI Whisper
- Converts raw English podcast audio into text
- Enables full end-to-end audio-to-Chinese processing

2. Abstractive Summarization
- Model: BART (facebook/bart-large-cnn)
- Generates both short and long summaries
- Focus: Length control and coherence preservation

3. Neural Machine Translation
- Model: MarianMT (Helsinki-NLP)
- Translates English summaries into Chinese
- Focus: Semantic fidelity and fluency

4. Audience-Aware Text Generation
- Model: FLAN-T5
- Rewrites content based on target audience: general public/students/professionals

Focus: Controlled text generation and content adaptation

5. Output Layer
Chinese text for:
- Short videos
- Long-form videos/podcasts
- Chinese audio narration

### System Dependency

FFmpeg is required for audio input mode.

Mac:
brew install ffmpeg

Ubuntu:
sudo apt update && sudo apt install ffmpeg

Windows:
Download from https://ffmpeg.org/download.html

## Web Demo (Streamlit)

Run the interactive demo:

```bash
streamlit run streamlit_app.py

conda install -c conda-forge ffmpeg

## Demo: English audio → Chinese audio (end-to-end)

1) Put a short English clip in the repo root as `sample_audio.mp3` (30–45s recommended).

2) Install dependencies:
```bash
pip install -r requirements.txt
