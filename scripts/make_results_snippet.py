import json, pandas as pd
from pathlib import Path

out_dir = Path("outputs/pilot")
rec = json.loads(out_dir.joinpath("run_records.jsonl").read_text(encoding="utf-8").splitlines()[-1])

lat = rec["latency_s"]
prox = rec.get("proxies", {})
cfg  = rec.get("config", {})

tex = []
tex.append(r"\subsection{Pilot results on one clip}")
tex.append(r"We ran InfluenceAgent end-to-end on a single English clip using Whisper-base, BART (CNN/DM), Opus-MT EN$\rightarrow$ZH, and a lightweight Chinese rewrite model.")
tex.append("")
tex.append(r"\paragraph{Latency breakdown.}")
tex.append(
    rf"On CPU, the end-to-end latency was {lat.get('e2e',0):.2f}s "
    rf"(ASR {lat.get('asr',0):.2f}s; Summ {lat.get('summ',0):.2f}s; MT {lat.get('mt',0):.2f}s; Rewrite {lat.get('rewrite',0):.2f}s)."
)
tex.append("")
tex.append(r"\paragraph{Proxy health checks.}")
tex.append(
    rf"ASR 3-gram repetition rate was {prox.get('asr_rep3',0):.3f}; "
    rf"summary compression ratio was {prox.get('sum_compression',0):.3f}; "
    rf"MT untranslated-span rate was {prox.get('mt_untrans_rate',0):.3f}. "
    rf"Rewrite structure compliance was {str(prox.get('rewrite_struct_ok',False))}."
)

# optional propagation report
prop_path = out_dir / "propagation_report.json"
if prop_path.exists():
    rep = json.loads(prop_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rep)
    # summarize drift at sev=0.10 if exists
    df10 = df[df["sev"]==0.10]
    if len(df10):
        tex.append("")
        tex.append(r"\paragraph{Propagation sensitivity.}")
        tex.append(
            rf"At 10\% perturbation severity, summary drift ranged "
            rf"{df10['drift_summ'].min():.3f}--{df10['drift_summ'].max():.3f}, "
            rf"translation drift {df10['drift_mt'].min():.3f}--{df10['drift_mt'].max():.3f}, "
            rf"and rewrite drift {df10['drift_rw'].min():.3f}--{df10['drift_rw'].max():.3f} "
            rf"(token-Jaccard distance)."
        )

snip = "\n".join(tex) + "\n"
(out_dir / "results_snippet.tex").write_text(snip, encoding="utf-8")
print("Wrote outputs/pilot/results_snippet.tex")
print("\n---\n" + snip + "---")
