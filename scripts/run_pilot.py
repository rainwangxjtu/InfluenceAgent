#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import whisper
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    gloss_model: str = "Helsinki-NLP/opus-mt-zh-en"
    bilingual_rewrite: bool = False
    whisper_model: str = "base"  # tiny/base/small/medium/large
    summ_model: str = "facebook/bart-large-cnn"
    mt_model: str = "Helsinki-NLP/opus-mt-en-zh"
    rewrite_model: str = "google/flan-t5-small"  # seq2seq OR causal (e.g., Qwen/Qwen2.5-0.5B-Instruct)

    device: str = "cpu"  # cpu/cuda/mps
    beams: int = 4

    max_new_tokens_sum: int = 120
    max_new_tokens_mt: int = 256
    max_new_tokens_rewrite: int = 220
    min_chars_rewrite: int = 80

    audience: str = "Chinese learners (student-friendly spoken script)"
    structured_rewrite: bool = True

    # prevent extremely long inputs from crashing generation
    max_input_chars_sum: int = 6000
    max_input_chars_mt: int = 4000
    max_input_chars_rewrite: int = 3500


# -----------------------------
# Helpers / utilities
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(device: str) -> torch.device:
    """
    Pin to the user's requested device. No auto device_map here.
    """
    device = (device or "cpu").lower().strip()
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] --device cuda requested but CUDA not available; falling back to cpu")
        return torch.device("cpu")
    if device == "mps":
        if is_mps_available():
            return torch.device("mps")
        print("[warn] --device mps requested but MPS not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device("cpu")


def clip_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...\n" + tail


def token_count_approx(text: str) -> int:
    # whitespace tokens for EN; chars for ZH-ish
    if not text:
        return 0
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return len(text)
    return len(text.split())


def compression_ratio(src: str, summ: str) -> float:
    src_n = max(token_count_approx(src), 1)
    summ_n = token_count_approx(summ)
    return float(summ_n) / float(src_n)


def repetition_rate_3gram(text: str) -> float:
    toks = text.split()
    if len(toks) < 3:
        return 0.0
    grams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    if not grams:
        return 0.0
    from collections import Counter

    c = Counter(grams)
    rep = sum(1 for _, k in c.items() if k > 1)
    return rep / max(len(c), 1)


def untranslated_span_rate_zh(text: str) -> float:
    if not text:
        return 0.0
    latin_or_digit = sum(ch.isascii() and (ch.isalpha() or ch.isdigit()) for ch in text)
    return latin_or_digit / max(len(text), 1)


def structure_compliance_zh(text: str) -> bool:
    # opening + 3 points + closing (simple marker heuristic)
    if not text or len(text.strip()) < 30:
        return False
    markers = ["第一", "第二", "第三", "1.", "2.", "3.", "1、", "2、", "3、"]
    count = sum(1 for m in markers if m in text)
    return count >= 3


def line_duplicate_rate(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    uniq = len(set(lines))
    return 1.0 - (uniq / len(lines))


def char_repeat_rate(text: str, span: int = 8) -> float:
    if not text or len(text) < span * 3:
        return 0.0
    seen = {}
    rep = 0
    total = 0
    for i in range(0, len(text) - span + 1):
        s = text[i : i + span]
        total += 1
        if s in seen:
            rep += 1
        else:
            seen[s] = i
    return rep / max(total, 1)


def chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return zh / max(len(text), 1)


def is_garbage_zh(text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Garbage detector for rewrite output.
    """
    meta: Dict[str, Any] = {}
    t = (text or "").strip()
    meta["len"] = len(t)

    if len(t) == 0:
        meta["reason"] = "empty"
        return True, meta

    if "<extra_id_" in t or re.search(r"<extra_id_\d+>", t):
        meta["reason"] = "sentinel_token"
        return True, meta

    zh_ratio = chinese_char_ratio(t)
    meta["zh_ratio"] = float(zh_ratio)
    if zh_ratio < 0.10:
        meta["reason"] = f"low_zh_ratio({zh_ratio:.2f})"
        return True, meta

    content_chars = sum(ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") for ch in t)
    content_ratio = content_chars / max(len(t), 1)
    meta["content_ratio"] = float(content_ratio)
    if len(t) < 10 or content_ratio < 0.20:
        meta["reason"] = f"too_low_content(len={len(t)}, content_ratio={content_ratio:.2f})"
        return True, meta

    ldr = line_duplicate_rate(t)
    cr8 = char_repeat_rate(t, span=8)
    meta["line_dup_rate"] = float(ldr)
    meta["char_rep8"] = float(cr8)

    if ldr > 0.45:
        meta["reason"] = f"high_line_dup_rate({ldr:.2f})"
        return True, meta
    if cr8 > 0.35:
        meta["reason"] = f"high_char_rep8({cr8:.2f})"
        return True, meta

    return False, meta


# -----------------------------
# Model loading + generation
# -----------------------------
def is_seq2seq_model(model_name: str) -> bool:
    """
    Decide if rewrite model should be loaded as seq2seq vs causal LM.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    # encoder-decoder models expose is_encoder_decoder=True
    return bool(getattr(cfg, "is_encoder_decoder", False))

def to_torch_device(device):
    """
    Accepts either:
      - string: "cpu" | "cuda" | "mps"
      - torch.device
    Returns torch.device.
    """
    import torch
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    # last resort (e.g., argparse namespace, None)
    return torch.device("cpu")

def load_seq2seq(model_name: str, device="cpu"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = to_torch_device(device)
    print(f"[load_seq2seq] {model_name} -> {device.type}")

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl = mdl.to(device)

    meta = {"kind": "seq2seq", "device": str(device)}
    return tok, mdl, meta


def load_causal(model_name: str, device="cpu"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = to_torch_device(device)
    print(f"[load_causal] {model_name} -> {device.type}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Pure CPU load (no device_map) to avoid accidental MPS usage
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    mdl = mdl.to(device)

    meta = {"kind": "causal", "device": str(device)}
    return tok, mdl, meta

def generate_seq2seq(
    tok,
    mdl,
    src_text: str,
    *,
    max_new_tokens: int,
    num_beams: int,
    min_new_tokens: int = 1,
    forced_bos_token_id: int | None = None,
) -> str:
    """
    Deterministic seq2seq generation with a non-empty guard.
    Note: forced_bos_token_id is useful for models like mBART/NLLB/M2M100,
    but should generally NOT be set for Marian (opus-mt-*) models.
    """
    device = next(mdl.parameters()).device
    inputs = tok(src_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        min_new_tokens=min_new_tokens,
    )

    # Only apply forced_bos_token_id when explicitly requested by caller
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)

    txt = tok.decode(out[0], skip_special_tokens=True).strip()
    return txt


def generate_causal(tok, mdl, prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
    device = next(mdl.parameters()).device

    # Use chat template if available (Qwen instruct has it)
    rendered = None
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(rendered, return_tensors="pt", truncation=True)
    else:
        inputs = tok(prompt, return_tensors="pt", truncation=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
        "eos_token_id": tok.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)

    decoded = tok.decode(out[0], skip_special_tokens=True)

    # If we used chat template, decoded includes the prompt; strip it if we can
    if rendered and decoded.startswith(rendered):
        decoded = decoded[len(rendered):]

    return decoded.strip()


# -----------------------------
# Pipeline stages
# -----------------------------
def asr_whisper(audio_path: str, whisper_model_name: str) -> Dict[str, Any]:
    model = whisper.load_model(whisper_model_name)
    result = model.transcribe(audio_path)
    return {"text": (result.get("text") or "").strip(), "language": result.get("language")}

def summarize_en(
    text_en: str,
    tok,
    mdl,
    *,
    max_new_tokens: int,
    num_beams: int = 4,
    max_input_chars: int = 8000,
) -> str:
    """
    Summarize English text using a pre-loaded seq2seq model (tok, mdl).
    IMPORTANT: This function assumes tok/mdl are already loaded.
    """
    if not text_en:
        return ""

    text_en = text_en.strip()
    if len(text_en) > max_input_chars:
        text_en = text_en[:max_input_chars]

    prompt = (
        "Summarize the following text in 2-4 sentences.\n\n"
        f"TEXT:\n{text_en}\n"
    )

    return generate_seq2seq(
        tok,
        mdl,
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

def translate_en2zh(
    text_en: str,
    tok,
    mdl,
    *,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    max_input_chars: int = 8000,
) -> str:
    """
    Translate English -> Chinese using a pre-loaded seq2seq model (tok, mdl).
    Marian (opus-mt-*) expects raw source text (NOT instruction prompts).
    Includes retry fallback if output is empty.
    """
    if not text_en:
        return ""

    text_en = text_en.strip()
    if len(text_en) > max_input_chars:
        text_en = text_en[:max_input_chars]

    model_type = getattr(getattr(mdl, "config", None), "model_type", "")
    is_marian = (model_type == "marian") or ("opus-mt" in str(getattr(mdl.config, "_name_or_path", "")).lower())

    # IMPORTANT: Marian wants raw text, not "Translate ..." prompts.
    src = text_en if is_marian else f"Translate English to Chinese:\n\n{text_en}\n"

    # SAFELY include forced_bos_token_id=0 for non-Marian models only.
    forced_bos = 0 if (not is_marian) else None

    zh = generate_seq2seq(
        tok,
        mdl,
        src,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        min_new_tokens=1,
        forced_bos_token_id=forced_bos,
    )

    # Retry fallback if still empty: force raw text + no forced_bos
    if not zh.strip():
        zh = generate_seq2seq(
            tok,
            mdl,
            text_en,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_new_tokens=1,
            forced_bos_token_id=None,
        )

    return zh.strip()


import re
import torch
from typing import Tuple, Dict, Any

def _is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"

def _contains_zh(text: str) -> bool:
    return any(_is_chinese_char(c) for c in text)

def _post_clean(text: str) -> str:
    if not text:
        return ""
    # remove common junk tokens / extra spaces
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

import re
from typing import Tuple, Dict, Any

def strip_prompt_echo(text: str) -> str:
    """
    Safer cleaner:
    - If model included an OUTPUT: block, keep only after it.
    - Otherwise, remove ONLY leading instruction-like lines until real content starts.
    - Do not aggressively drop by prefix across the whole output.
    """
    if not text:
        return ""
    t = text.strip()

    # If model echoed an OUTPUT section, keep only after OUTPUT:
    m = re.search(r"(?im)^\s*OUTPUT\s*:\s*$", t)
    if m:
        t = t[m.end():].strip()

    bad_leading_prefixes = (
        "硬性格式要求", "要求", "规则", "Rules", "INPUT", "OUTPUT",
        "内容：", "内容:", "只输出", "请把下面", "Convert the following",
        "For EACH line",
    )

    lines = [ln.rstrip() for ln in t.splitlines()]
    kept = []
    started = False

    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        if not started:
            # skip only obvious prompt-y lead lines
            if any(s.startswith(p) for p in bad_leading_prefixes):
                continue
            # also skip bilingual template headers if they appear
            if s.startswith("ZH:") or s.startswith("EN:"):
                continue
            started = True

        kept.append(ln)

    t = "\n".join(kept).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t



def has_struct_markers(text: str) -> bool:
    """Check required structure markers exist."""
    if not text:
        return False
    return ("第一" in text) and ("第二" in text) and ("第三" in text)

def rewrite_zh_to_audience(
    text_zh: str,
    tok,
    mdl,
    audience: str,
    structured: bool,
    max_new_tokens: int,
    num_beams: int,
    min_chars: int,
    model_kind: str = "seq2seq",   # "seq2seq" or "causal"
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "used_fallback": False,
        "structured": structured,
        "model_kind": model_kind,
    }

    text_zh = (text_zh or "").strip()
    if not text_zh:
        meta["failure_reason"] = "empty_input"
        return "", meta

    def has_struct_markers(t: str) -> bool:
        return ("第一" in t) and ("第二" in t) and ("第三" in t)

    def looks_like_copy(inp: str, out: str) -> bool:
        # crude but useful: output equals input or nearly equals input
        inp_s = re.sub(r"\s+", "", inp)
        out_s = re.sub(r"\s+", "", out)
        if not out_s:
            return True
        if out_s == inp_s:
            return True
        # if output is >80% identical prefix-wise, treat as copy
        common = 0
        for a, b in zip(inp_s, out_s):
            if a == b:
                common += 1
            else:
                break
        return (common / max(len(out_s), 1)) > 0.80

    def run_once(prompt: str) -> str:
        if model_kind == "causal":
            return generate_causal_only(tok, mdl, prompt, max_new_tokens=max_new_tokens)
        else:
            return generate_seq2seq(tok, mdl, prompt, max_new_tokens=max_new_tokens, num_beams=num_beams, min_new_tokens=1)

    if structured:
        prompts = [
            # attempt 1: clear constraints
            (
                f"请把下面中文内容改写成适合 {audience} 的口语讲稿。\n"
                "只输出讲稿正文，不要输出任何提示语/规则/标题。\n"
                "格式必须严格包含：\n"
                "1) 开头一句话点明主题。\n"
                "2) 用“第一：”“第二：”“第三：”三点说明（必须出现这三个标记）。\n"
                "3) 最后一句话总结。\n"
                "4) 不要原封不动复述原文，要用更口语的表达。\n"
                "\n"
                f"{text_zh}\n"
            ),
            # attempt 2: forbid copying + shorten sentences
            (
                f"把下面中文改写成更自然的口语讲稿，适合 {audience}。\n"
                "硬性要求：必须包含“第一：”“第二：”“第三：”和最后总结一句；每句尽量短。\n"
                "禁止：复制原句、输出规则、输出“内容/INPUT/OUTPUT”。\n"
                "\n"
                f"{text_zh}\n"
            ),
            # attempt 3: “模板化”约束
            (
                f"输出一个口语讲稿（适合 {audience}），严格按以下模板输出：\n"
                "【主题句】…\n"
                "第一：…\n"
                "第二：…\n"
                "第三：…\n"
                "【总结句】…\n"
                "禁止复述原文句子，必须改写。\n"
                "\n"
                f"{text_zh}\n"
            ),
        ]
    else:
        prompts = [
            (
                f"请把下面中文内容改写得更口语、更适合 {audience}。\n"
                "只输出改写后的文本，不要输出任何提示语。\n"
                "不要原封不动复述原文。\n"
                f"{text_zh}\n"
            )
        ]

    best = ""
    for i, p in enumerate(prompts, start=1):
        out = run_once(p).strip()
        out = strip_prompt_echo(out)

        bad = (
            (not out) or
            (len(out) < min_chars) or
            ("请把下面" in out[:80]) or
            ("只输出" in out[:80]) or
            ("内容" in out[:80]) or
            ("INPUT" in out[:80].upper()) or
            ("OUTPUT" in out[:80].upper()) or
            ("Rules" in out[:80]) or
            (structured and not has_struct_markers(out)) or
            looks_like_copy(text_zh, out)
        )

        if not bad:
            best = out
            break

        best = out  # keep latest attempt for debugging
        meta["used_fallback"] = True
        meta["attempt_failed"] = i

    if structured and (not has_struct_markers(best)):
        meta["failure_reason"] = "missing_structure_markers_or_copy"
        # return "", meta   # uncomment if you prefer hard-fail
    return best, meta


import re
from typing import List

def split_zh_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln]

def make_bilingual_gloss(
    zh_text: str,
    tok_zh_en,
    mdl_zh_en,
    *,
    max_new_tokens: int = 80,
    num_beams: int = 4,
) -> str:
    """
    Create aligned bilingual output:
      ZH: ...
      EN: ...
    per Chinese line, using a zh->en seq2seq model.
    """
    lines = split_zh_lines(zh_text)
    if not lines:
        return ""

    blocks = []
    for ln in lines:
        # translate each line separately to keep alignment
        en = generate_seq2seq(
            tok_zh_en,
            mdl_zh_en,
            ln,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        en = re.sub(r"\s+", " ", (en or "").strip())
        blocks.append(f"ZH: {ln}\nEN: {en}")

    return "\n\n".join(blocks).strip()

import re
from typing import List, Tuple, Dict, Any

def is_causal_model_name(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ["qwen", "llama", "mistral", "gemma", "gpt", "phi"])

def generate_causal_only(tok, mdl, user_prompt: str, max_new_tokens: int = 256) -> str:
    """
    For chat/instruct causal LMs (e.g., Qwen). Returns ONLY newly generated text.
    Uses chat template if available; slices off the prompt tokens.
    """
    device = next(mdl.parameters()).device

    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You rewrite Chinese text for language learners. Follow format constraints exactly."},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(rendered, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
    else:
        inputs = tok(user_prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        )

    gen_ids = out[0, prompt_len:]  # slice off prompt
    txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return txt

# -----------------------------
# Main runner
# -----------------------------
def run_one_clip(audio_path: str, out_dir: Path, cfg: RunConfig) -> Dict[str, Any]:
    clip_id = Path(audio_path).stem
    device = resolve_device(cfg.device)

    record: Dict[str, Any] = {
        "clip_id": clip_id,
        "audio_path": str(audio_path),
        "config": asdict(cfg),
        "timestamps_ms": {},
        "latency_s": {},
        "texts": {},
        "proxies": {},
        "notes": {},
    }
    safe_mkdir(out_dir)

    # Stage 1: ASR
    t0 = time.time()
    record["timestamps_ms"]["asr_start"] = now_ms()
    asr = asr_whisper(audio_path, cfg.whisper_model)
    record["timestamps_ms"]["asr_end"] = now_ms()
    record["latency_s"]["asr"] = time.time() - t0
    a = asr["text"]
    record["texts"]["asr_en"] = a
    record["notes"]["asr_lang"] = asr.get("language")

    # --- Preload downstream models once per clip (fast + avoids reload in perturb script) ---
    tok_s, mdl_s, _ = load_seq2seq(cfg.summ_model, cfg.device)
    tok_m, mdl_m, _ = load_seq2seq(cfg.mt_model, cfg.device)

    # Stage 2: Summarize (English)
    t0 = time.time()
    record["timestamps_ms"]["summ_start"] = now_ms()
    s = summarize_en(
        a,
        tok_s,
        mdl_s,
        max_new_tokens=cfg.max_new_tokens_sum,
        num_beams=cfg.beams,
        max_input_chars=8000,
    )
    record["timestamps_ms"]["summ_end"] = now_ms()
    record["latency_s"]["summ"] = time.time() - t0
    record["texts"]["sum_en"] = s

    # Stage 3: Translate (EN->ZH)
    t0 = time.time()
    record["timestamps_ms"]["mt_start"] = now_ms()
    z = translate_en2zh(
        s,
        tok_m,
        mdl_m,
        max_new_tokens=256,
        num_beams=cfg.beams,
        max_input_chars=8000,
    )
    if not z.strip():
        print("[error] mt_zh is empty. sum_en preview:", s[:160].replace("\n"," "))
    record["timestamps_ms"]["mt_end"] = now_ms()
    record["latency_s"]["mt"] = time.time() - t0
    record["texts"]["mt_zh"] = z

    # -----------------------------
    # Stage 4: Rewrite (ZH -> audience)
    # -----------------------------
    t0 = time.time()
    record["timestamps_ms"]["rewrite_start"] = now_ms()

    rewrite_is_causal = any(
        k in (cfg.rewrite_model or "").lower()
        for k in ["qwen", "llama", "mistral", "gemma", "gpt", "phi"]
    )

    if rewrite_is_causal:
        tok_rw, mdl_rw, rw_meta = load_causal(cfg.rewrite_model, device=cfg.device)
    else:
        tok_rw, mdl_rw, rw_meta = load_seq2seq(cfg.rewrite_model, device=cfg.device)

    rw_kind = rw_meta["kind"]  # "causal" or "seq2seq"

    if not z or not z.strip():
        record["notes"]["failure_reason"] = "mt_zh_empty"
        record["texts"]["rewrite_zh"] = ""
        record["texts"]["rewrite_bilingual"] = ""
    # still save upstream artifacts for debugging
        safe_mkdir(out_dir)
        (out_dir / f"{clip_id}.asr_en.txt").write_text(a or "", encoding="utf-8")
        (out_dir / f"{clip_id}.sum_en.txt").write_text(s or "", encoding="utf-8")
        (out_dir / f"{clip_id}.mt_zh.txt").write_text(z or "", encoding="utf-8")
        return record

    r, rw_meta2 = rewrite_zh_to_audience(
        z,
        tok_rw,
        mdl_rw,
        audience=cfg.audience,
        structured=cfg.structured_rewrite,
        max_new_tokens=cfg.max_new_tokens_rewrite,
        num_beams=cfg.beams,
        min_chars=cfg.min_chars_rewrite,
        model_kind=rw_kind,
    )

    record["notes"]["rewrite_meta"] = rw_meta2


    record["timestamps_ms"]["rewrite_end"] = now_ms()
    record["latency_s"]["rewrite"] = time.time() - t0
    record["texts"]["rewrite_zh"] = r
    record["notes"]["rewrite_meta"] = rw_meta

    # Save Chinese rewrite
    (out_dir / f"{clip_id}.rewrite_zh.txt").write_text(r or "", encoding="utf-8")

        # -----------------------------
    # Optional: bilingual gloss (ZH + EN) built from FINAL r
    # -----------------------------
    if cfg.bilingual_rewrite and (r or "").strip():
        tok_zh_en, mdl_zh_en, _ = load_seq2seq(cfg.gloss_model, device=cfg.device)
        r_bi = make_bilingual_gloss(
            r,
            tok_zh_en,
            mdl_zh_en,
            max_new_tokens=80,
            num_beams=cfg.beams,
        )
        record["texts"]["rewrite_bilingual"] = r_bi
        (out_dir / f"{clip_id}.rewrite_bilingual.txt").write_text(r_bi or "", encoding="utf-8")

    # End-to-end latency (fixes your KeyError)
    record["latency_s"]["e2e"] = (
        record["latency_s"].get("asr", 0.0)
        + record["latency_s"].get("summ", 0.0)
        + record["latency_s"].get("mt", 0.0)
        + record["latency_s"].get("rewrite", 0.0)
    )

    # Proxies
    record["proxies"]["asr_rep3"] = repetition_rate_3gram(a)
    record["proxies"]["sum_compression"] = compression_ratio(a, s)
    record["proxies"]["mt_untrans_rate"] = untranslated_span_rate_zh(z)
    record["proxies"]["rewrite_struct_ok"] = structure_compliance_zh(r)

    g, gmeta = is_garbage_zh(r)
    record["proxies"]["rewrite_garbage"] = bool(g)
    record["proxies"]["rewrite_line_dup_rate"] = float(gmeta.get("line_dup_rate", 0.0))
    record["proxies"]["rewrite_char_rep8"] = float(gmeta.get("char_rep8", 0.0))
    record["proxies"]["rewrite_zh_ratio"] = float(gmeta.get("zh_ratio", chinese_char_ratio(r)))

    # Save artifacts
    safe_mkdir(out_dir)
    (out_dir / f"{clip_id}.asr_en.txt").write_text(a, encoding="utf-8")
    (out_dir / f"{clip_id}.sum_en.txt").write_text(s, encoding="utf-8")
    (out_dir / f"{clip_id}.mt_zh.txt").write_text(z, encoding="utf-8")
    (out_dir / f"{clip_id}.rewrite_zh.txt").write_text(r, encoding="utf-8")
    
    return record

def write_summary(run_records: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.json_normalize(run_records)

    # Ensure e2e column exists even if something changes upstream
    stage_cols = [f"latency_s.{s}" for s in ["asr", "summ", "mt", "rewrite"] if f"latency_s.{s}" in df.columns]
    if "latency_s.e2e" not in df.columns and stage_cols:
        df["latency_s.e2e"] = df[stage_cols].sum(axis=1)

    # Quick summary CSV
    cols = [
        "clip_id",
        "latency_s.asr",
        "latency_s.summ",
        "latency_s.mt",
        "latency_s.rewrite",
        "latency_s.e2e",
        "proxies.asr_rep3",
        "proxies.sum_compression",
        "proxies.mt_untrans_rate",
        "proxies.rewrite_struct_ok",
        "proxies.rewrite_garbage",
        "proxies.rewrite_line_dup_rate",
        "proxies.rewrite_char_rep8",
        "notes.rewrite_meta.used_fallback",
        "notes.rewrite_meta.failure_reason",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_dir / "pilot_summary.csv", index=False)

    # Timing table JSON
    def p(x, q):
        return float(np.quantile(x, q)) if len(x) else float("nan")

    timing: Dict[str, Any] = {}
    for k in ["asr", "summ", "mt", "rewrite", "e2e"]:
        col = f"latency_s.{k}"
        if col not in df.columns:
            continue
        xs = df[col].dropna().values
        timing[k] = {
            "mean": float(np.mean(xs)) if len(xs) else float("nan"),
            "p90": p(xs, 0.90),
            "p95": p(xs, 0.95),
        }

    (out_dir / "timing_table.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilingual", action="store_true",
                    help="Rewrite output as bilingual: Chinese + English gloss")
    ap.add_argument("--audio", required=True, help="Path to one audio file (mp3/wav)")
    ap.add_argument("--out", default="outputs/pilot", help="Output dir")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--audience", default="Chinese learners (student-friendly spoken script)")
    ap.add_argument("--structured", action="store_true", help="Use structured rewrite prompt (opening + 3 points + closing)")
    

    # model overrides
    ap.add_argument("--whisper_model", default="base")
    ap.add_argument("--summ_model", default="facebook/bart-large-cnn")
    ap.add_argument("--mt_model", default="Helsinki-NLP/opus-mt-en-zh")
    ap.add_argument("--rewrite_model", default="google/flan-t5-small")
    ap.add_argument("--gloss_model", default="Helsinki-NLP/opus-mt-zh-en")

    # generation params
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_new_tokens_sum", type=int, default=120)
    ap.add_argument("--max_new_tokens_mt", type=int, default=256)
    ap.add_argument("--max_new_tokens_rewrite", type=int, default=220)
    ap.add_argument("--min_chars_rewrite", type=int, default=80)

    args = ap.parse_args()

    cfg = RunConfig(
        whisper_model=args.whisper_model,
        summ_model=args.summ_model,
        mt_model=args.mt_model,
        rewrite_model=args.rewrite_model,
        device=args.device,
        beams=args.beams,
        max_new_tokens_sum=args.max_new_tokens_sum,
        max_new_tokens_mt=args.max_new_tokens_mt,
        max_new_tokens_rewrite=args.max_new_tokens_rewrite,
        min_chars_rewrite=args.min_chars_rewrite,
        audience=args.audience,
        structured_rewrite=bool(args.structured),
        bilingual_rewrite=bool(args.bilingual)
    )

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    record = run_one_clip(args.audio, out_dir, cfg)

    # Append JSONL
    jsonl_path = out_dir / "run_records.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary files
    write_summary([record], out_dir)

    print("✅ Pilot finished.")
    print(f"- run_records: {jsonl_path}")
    print(f"- summary CSV: {out_dir / 'pilot_summary.csv'}")
    print(f"- timing JSON: {out_dir / 'timing_table.json'}")
    print(f"- texts saved: {out_dir}")


if __name__ == "__main__":
    main()