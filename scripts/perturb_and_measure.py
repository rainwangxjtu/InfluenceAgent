import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

import scripts.run_pilot as rp


def read_first_record(jsonl_path: Path):
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"Empty JSONL: {jsonl_path}")
    return json.loads(lines[0])


def delete_tokens(tokens, sev, rng):
    # delete ~sev fraction of tokens
    n = len(tokens)
    if n == 0:
        return tokens
    k = max(1, int(round(sev * n)))
    idx = set(rng.sample(range(n), k=min(k, n)))
    return [t for i, t in enumerate(tokens) if i not in idx]


def jaccard_tok(a: str, b: str) -> float:
    A = set(a.split())
    B = set(b.split())
    if not A and not B:
        return 0.0
    return 1.0 - (len(A & B) / max(len(A | B), 1))


def char_bigrams(s: str):
    s = s.strip()
    return set(s[i : i + 2] for i in range(len(s) - 1)) if len(s) >= 2 else set()


def jaccard_bigram(a: str, b: str) -> float:
    A = char_bigrams(a)
    B = char_bigrams(b)
    if not A and not B:
        return 0.0
    return 1.0 - (len(A & B) / max(len(A | B), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="run_records.jsonl from run_pilot")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--severities", default="0.05,0.10,0.20", help="comma-separated severities")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec0 = read_first_record(in_path)
    a0 = rec0.get("texts", {}).get("asr_en", "")
    if not a0:
        raise SystemExit("No baseline ASR text found in run_records.jsonl (texts.asr_en is empty).")

    # Rebuild config (robust to extra keys)
    cfg_dict = rec0.get("config", {})
    cfg_fields = set(getattr(rp.RunConfig, "__annotations__", {}).keys())
    cfg = rp.RunConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_fields})

    cpu = torch.device("cpu")
    rng = random.Random(args.seed)

    # Load models once
    tok_s, mdl_s, *_ = rp.load_seq2seq(cfg.summ_model, cpu)
    tok_m, mdl_m, *_ = rp.load_seq2seq(cfg.mt_model, cpu)

    # Rewrite model: prefer causal (Qwen) if available; else seq2seq
    tok_rw = mdl_rw = None
    rw_kind = "seq2seq"
    if hasattr(rp, "load_causal") and ("Qwen" in cfg.rewrite_model or "Instruct" in cfg.rewrite_model):
        tok_rw, mdl_rw, _ = rp.load_causal(cfg.rewrite_model, cpu)
        rw_kind = "causal"
    else:
        tok_rw, mdl_rw = rp.load_seq2seq(cfg.rewrite_model, cpu)

    # baseline downstream
    s0 = rp.summarize_en(a0, tok_s, mdl_s, max_new_tokens=cfg.max_new_tokens_sum, num_beams=cfg.beams, max_input_chars=8000)
    z0 = rp.translate_en2zh(s0, tok_m, mdl_m, max_new_tokens=256, num_beams=cfg.beams, max_input_chars=8000)
    r0, meta0 = rp.rewrite_zh_to_audience(
        z0, tok_rw, mdl_rw,
        audience=cfg.audience,
        structured=cfg.structured_rewrite,
        max_new_tokens=cfg.max_new_tokens_rewrite,
        num_beams=cfg.beams,
        min_chars=cfg.min_chars_rewrite,
        bilingual=True,   # <-- if your function supports it; safe to remove if not
        model_kind=rw_kind,  # <-- safe to remove if not used in your implementation
    )

    # severities
    severities = [float(x) for x in args.severities.split(",") if x.strip()]
    tokens = a0.split()

    report = []
    for sev in severities:
        a_del = " ".join(delete_tokens(tokens, sev, rng))

        s = rp.summarize_en(a_del, tok_s, mdl_s, max_new_tokens=cfg.max_new_tokens_sum, num_beams=cfg.beams, max_input_chars=8000)
        z = rp.translate_en2zh(s, tok_m, mdl_m, max_new_tokens=256, num_beams=cfg.beams, max_input_chars=8000)
        r, meta = rp.rewrite_zh_to_audience(
            z, tok_rw, mdl_rw,
            audience=cfg.audience,
            structured=cfg.structured_rewrite,
            max_new_tokens=cfg.max_new_tokens_rewrite,
            num_beams=cfg.beams,
            min_chars=cfg.min_chars_rewrite,
            bilingual=True,      # remove if unsupported
            model_kind=rw_kind,  # remove if unsupported
        )

        row = {
            "perturb": "del",
            "sev": sev,
            "drift_summ_tokJ": jaccard_tok(s0, s),
            "drift_mt_bigramJ": jaccard_bigram(z0, z),
            "drift_rw_bigramJ": jaccard_bigram(r0, r),
            "rw_struct_ok": rp.structure_compliance_zh(r),
            "rw_garbage": bool(rp.is_garbage_zh(r)[0]) if hasattr(rp, "is_garbage_zh") else False,
            "rw_used_fallback": meta.get("used_fallback", False) if isinstance(meta, dict) else False,
        }
        report.append(row)

    df = pd.DataFrame(report)
    df.to_csv(out_dir / "perturb_report.csv", index=False)
    (out_dir / "perturb_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # also dump baseline artifacts for inspection
    (out_dir / "baseline_asr_en.txt").write_text(a0, encoding="utf-8")
    (out_dir / "baseline_sum_en.txt").write_text(s0, encoding="utf-8")
    (out_dir / "baseline_mt_zh.txt").write_text(z0, encoding="utf-8")
    (out_dir / "baseline_rewrite.txt").write_text(r0, encoding="utf-8")

    print("✅ perturb_and_measure finished.")
    print(f"- wrote: {out_dir/'perturb_report.csv'}")
    print(f"- wrote: {out_dir/'perturb_report.json'}")


if __name__ == "__main__":
    main()
