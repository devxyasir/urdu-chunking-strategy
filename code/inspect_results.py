import json
import os
from collections import Counter

import numpy as np

D = "data/external/urdu_chunking_results"
lines = []

lines.append("## RESULTS_README")
lines.append(open(os.path.join(D, "RESULTS_README.md"), encoding="utf-8").read())

m = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
lines.append("\n## manifest keys")
lines.append(json.dumps(list(m.keys()), ensure_ascii=False))
lines.append(json.dumps(m, ensure_ascii=False)[:3000])

for name in ["aux_.json", "tokenizer_fertility.json"]:
    lines.append("\n## " + name)
    lines.append(open(os.path.join(D, name), encoding="utf-8").read()[:2000])

import csv

with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))
lines.append(f"\n## questions.csv rows {len(qs)}")
lines.append(json.dumps(qs[0], ensure_ascii=False)[:500])
lines.append(str(Counter(q.get("split") for q in qs)))

for name in ["corpus_ur.jsonl", "corpus_en.jsonl"]:
    recs = [json.loads(l) for l in open(os.path.join(D, name), encoding="utf-8")]
    lines.append(f"\n## {name}: {len(recs)} records")
    r = recs[0]
    lines.append("keys " + str(list(r.keys())))
    for k, v in r.items():
        if isinstance(v, str):
            lines.append(f"  {k}: str[{len(v)}]")
        elif isinstance(v, list):
            lines.append(f"  {k}: list[{len(v)}] first={str(v[0])[:120] if v else ''}")
        else:
            lines.append(f"  {k}: {v}")

runs = [json.loads(l) for l in open(os.path.join(D, "runs.jsonl"), encoding="utf-8")]
lines.append(f"\n## runs.jsonl: {len(runs)} runs")
lines.append("keys: " + str(list(runs[0].keys())))
for r in runs[:5]:
    pq = r.get("per_question") or {}
    lens = {k: len(v) for k, v in pq.items() if isinstance(v, list)}
    lines.append(f"- {r.get('run_id')} corpus={r.get('corpus')} retriever={r.get('retriever')} "
                 f"strategy={r.get('strategy')} size={r.get('size')} ov={r.get('overlap')} "
                 f"chunks={r.get('chunks')} truncated={r.get('truncated')} lens={lens}")
lines.append("\nall run ids:")
for r in runs:
    lines.append(f"{r.get('run_id')} | {r.get('corpus')} | {r.get('retriever')} | {r.get('model')} | "
                 f"{r.get('strategy')} | {r.get('size')} | {r.get('overlap')} | {r.get('normalization')}")

idx = [json.loads(l) for l in open(os.path.join(D, "run_index.jsonl"), encoding="utf-8")]
lines.append(f"\n## run_index.jsonl: {len(idx)} rows")
lines.append(json.dumps(idx[:3], ensure_ascii=False)[:1500])

for name in ["rankings_B.npz", "rankings_C.npz", "rankings_E.npz"]:
    z = np.load(os.path.join(D, name), allow_pickle=True)
    lines.append(f"\n## {name}: {list(z.keys())[:20]}")
    for k in list(z.keys())[:6]:
        a = z[k]
        lines.append(f"  {k}: shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}")

st = json.load(open(os.path.join(D, "state.json"), encoding="utf-8"))
lines.append("\n## state.json")
lines.append(json.dumps(st, ensure_ascii=False)[:3000])

open("results/analysis/inspect_results.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
