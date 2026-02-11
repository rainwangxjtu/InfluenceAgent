import json, numpy as np, pandas as pd
from pathlib import Path

p = Path("outputs/pilot_repeat/run_records.jsonl")
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.json_normalize(rows)

def q(x, qq): return float(np.quantile(np.asarray(x, dtype=float), qq))

lat_cols = ["asr","summ","mt","rewrite","e2e"]
table = []
for k in lat_cols:
    col = f"latency_s.{k}"
    if col not in df.columns: 
        continue
    xs = df[col].astype(float).tolist()
    table.append({
        "stage": k,
        "mean": float(np.mean(xs)),
        "p50": q(xs, 0.50),
        "p90": q(xs, 0.90),
        "p95": q(xs, 0.95),
    })

out = pd.DataFrame(table)
print(out.to_string(index=False))

(out).to_csv("outputs/pilot_repeat/timing_agg.csv", index=False)
print("\nWrote outputs/pilot_repeat/timing_agg.csv")
