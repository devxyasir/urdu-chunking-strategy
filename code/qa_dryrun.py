import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

D = "data/external/urdu_chunking_results"
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
runs = [json.loads(l) for l in open(os.path.join(D, "runs.jsonl"), encoding="utf-8")]
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))
qsplit = Counter(q["split"] for q in qs)

lines = ["# Dry-run QA (NOT study results)", "",
         f"- manifest dry_run: {man.get('dry_run')}",
         f"- GPU: {man.get('gpu')}",
         f"- runs: {len(runs)}",
         f"- questions: {len(qs)} ({dict(qsplit)})",
         f"- corpus_ur records: {sum(1 for _ in open(os.path.join(D, 'corpus_ur.jsonl'), encoding='utf-8'))}",
         f"- corpus_en records: {sum(1 for _ in open(os.path.join(D, 'corpus_en.jsonl'), encoding='utf-8'))}",
         f"- self-tests: {sum(1 for v in man['self_tests'].values() if v['pass'])}/{len(man['self_tests'])} passed",
         f"- runtime deviations: {man.get('deviations_runtime')}",
         ""]

bad = []
for r in runs:
    n = qsplit[r["split"]]
    for k, v in r["per_question"].items():
        if isinstance(v, list) and len(v) != n:
            bad.append(f"{r['run_id']} {k} len={len(v)} expected={n}")
lines.append(f"## per_question length check")
lines.append(f"runs with any wrong-length array: {len({b.split()[0] for b in bad})}")
for b in bad[:10]:
    lines.append("- " + b)

rk = np.load(os.path.join(D, "rankings_B.npz"), allow_pickle=True)
ids = list(rk["run_ids"])
lines.append("")
lines.append(f"## rankings_B: {len(ids)} runs")
mism = 0
for rid in ids:
    key = f"{rid}|top10"
    if key not in rk:
        mism += 1
        continue
    if rk[key].shape[0] != qsplit["test"]:
        mism += 1
lines.append(f"top10 rows misaligned: {mism}")

lines.append("")
lines.append("## Run inventory")
lines.append(str(Counter((r["retriever"], r["model"]) for r in runs)))

lines.append("")
lines.append("## Smoke numbers (40 test questions, 1 article; NOT findings)")
lines.append("| run | bm500 | bm500x | r10 | mrr | chunks |")
lines.append("| --- | --- | --- | --- | --- | --- |")
for r in runs:
    if r["retriever"] in ("dense", "rrf") or (r["retriever"] == "bm25_ws" and r["strategy"] == "fixed_words" and r["size"] in (50, 100, 200) and r["overlap"] == 0.0 and r["normalization"] == "N0"):
        m = r["metrics"]
        lines.append(f"| {r['run_id']} | {m['bm500']:.3f} | {m.get('bm500x', float('nan')):.3f} | {m['r10']:.3f} | {m['mrr10']:.3f} | {r['chunks']} |")

lines.append("")
lines.append("## What is missing for the study")
lines.append("- Full 150-article / 120-test-article corpus (currently 2 articles, 1 test).")
lines.append("- Questions: 40 test (spec: 7,362).")
lines.append("- Dense models: only bge-m3, only one config (fixed_words 100). Need e5, qwen, labse and 21 configs each.")
lines.append("- English control: absent. Native set: absent. Full BM25 grid on full corpus: partial by construction of the dry run.")
lines.append("- Article-level CIs are impossible with 1 test article.")

open("results/analysis/dryrun_qa.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
