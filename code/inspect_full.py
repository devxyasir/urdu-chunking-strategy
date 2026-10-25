import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

D = "data/external/urdu_chunking_full"
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
idx = [json.loads(l) for l in open(os.path.join(D, "run_index.jsonl"), encoding="utf-8")]
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))

lines = ["# Full-run inspection", ""]
lines.append("## Manifest")
for k in ["schema_version", "timestamp", "dry_run", "gpu", "torch", "cuda", "sentence_transformers", "n_runs"]:
    lines.append(f"- {k}: {man.get(k)}")
lines.append(f"- actual_counts: {json.dumps(man.get('actual_counts'), ensure_ascii=False)}")
lines.append(f"- expected_counts: {json.dumps(man.get('expected_counts'), ensure_ascii=False)}")
lines.append(f"- model_revisions: {json.dumps(man.get('model_revisions'), ensure_ascii=False)}")
lines.append(f"- tokenizer_fertility: {json.dumps(man.get('tokenizer_fertility'), ensure_ascii=False)}")
lines.append(f"- prefix_checks: {json.dumps(man.get('prefix_checks'), ensure_ascii=False)}")
lines.append(f"- deviations_runtime: {json.dumps(man.get('deviations_runtime'), ensure_ascii=False)}")
lines.append(f"- data_stats: {json.dumps(man.get('data_stats'), ensure_ascii=False)}")
lines.append(f"- normalization_mapping: {json.dumps(man.get('normalization_mapping'), ensure_ascii=False)}")
lines.append("")
lines.append("## Self-tests")
for k, v in man.get("self_tests", {}).items():
    lines.append(f"- {'PASS' if v.get('pass') else 'FAIL'} | {k} | {v.get('detail','')[:300]}")

lines.append("")
lines.append("## Questions")
cnt = Counter(q["split"] for q in qs)
lines.append(f"- rows: {len(qs)} | {dict(cnt)}")
lines.append(f"- first: {json.dumps({k: qs[0][k] for k in list(qs[0])[:8]}, ensure_ascii=False)[:300]}")

lines.append("")
lines.append("## Runs inventory")
lines.append(f"- runs in index: {len(idx)}")
lines.append(f"- by retriever: {dict(Counter(r['retriever'] for r in idx))}")
lines.append(f"- by model: {dict(Counter(str(r.get('model')) for r in idx))}")
lines.append(f"- by stage: {dict(Counter(r.get('stage') for r in idx))}")
lines.append(f"- by corpus/split: {dict(Counter((r['corpus'], r['split']) for r in idx))}")
lines.append(f"- truncated runs: {sum(1 for r in idx if r.get('truncated'))}")
tr = [r for r in idx if r.get("truncated")]
lines.append(f"- truncated ids: {[r['run_id'] for r in tr][:40]}")

lines.append("")
lines.append("## Token budgets present")
lines.append(str(sorted({r["token_budget"] for r in idx if r.get("token_budget")})))
lines.append("## Dense strategies present per model")
for m in sorted({r.get("model") for r in idx if r.get("model")}):
    c = Counter(r["strategy"] for r in idx if r.get("model") == m and r["retriever"] == "dense")
    lines.append(f"- {m}: {dict(c)}")

lines.append("")
lines.append("## Truncation rates (dense, by model and size)")
for m in ["BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct", "Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/LaBSE"]:
    rows = [r for r in idx if r.get("model") == m and r["retriever"] == "dense"]
    if not rows:
        continue
    for size in [50, 100, 200, 400, 800]:
        rr = [r for r in rows if r.get("size") == size and r.get("overlap") == 0.0 and r["strategy"] == "fixed_words"]
        if rr:
            lines.append(f"- {m.split('/')[-1]} {size}w: trunc={rr[0]['truncation_rate']:.4f} chunks={rr[0]['chunks']}")

lines.append("")
lines.append("## Rankings")
for f in ["rankings_B.npz", "rankings_C.npz", "rankings_D.npz", "rankings_E.npz"]:
    z = np.load(os.path.join(D, f), allow_pickle=True)
    ids = list(z["run_ids"])
    bad = 0
    for rid in ids:
        k = f"{rid}|top10"
        if k not in z:
            bad += 1
            continue
        n = cnt["test"] if "test" in rid else (cnt["dev"] if "dev" in rid else len(qs))
        if z[k].shape[0] != n:
            bad += 1
    lines.append(f"- {f}: {len(ids)} runs, misaligned {bad}")

lines.append("")
lines.append("## Aggregate metrics (test, N0, overlap 0) — bm500x")
def get(r):
    return r["metrics"].get("bm500x", r["metrics"].get("bm500"))

for corpus in ["uqa_ur", "squad_en", "native_ur"]:
    for m in ["none", "BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct", "Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/LaBSE"]:
        rows = [r for r in idx if r["corpus"] == corpus and r.get("model") == (None if m == "none" else m) and r["retriever"] in ("bm25_ws", "dense")]
        if not rows:
            continue
        lines.append(f"### {corpus} / {m}")
        for strat in ["fixed_words", "fixed_tokens", "sentence_aware", "recursive_english", "recursive_urdu"]:
            rr = [r for r in rows if r["strategy"] == strat and r.get("overlap") == 0.0 and (r.get("size") in (50, 100, 200) or r.get("token_budget") in (64, 128, 256))]
            if not rr:
                continue
            vals = []
            for r in sorted(rr, key=lambda x: (x.get("size") or x.get("token_budget"))):
                key = r.get("size") or f"{r.get('token_budget')}t"
                vals.append(f"{key}:{get(r):.3f}")
            lines.append(f"- {strat}: " + " ".join(vals))

open("results/analysis/full_inspect.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
