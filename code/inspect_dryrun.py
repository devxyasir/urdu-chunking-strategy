import json
import os

D = "data/external/urdu_chunking_results"
m = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
lines = ["## settings", json.dumps(m.get("settings"), indent=1, ensure_ascii=False),
         "## deviations_runtime", json.dumps(m.get("deviations_runtime"), indent=1, ensure_ascii=False),
         "## dry_run", m.get("dry_run"), "## n_runs", m.get("n_runs"),
         "## actual_counts", json.dumps(m.get("actual_counts"), ensure_ascii=False)]

runs = [json.loads(l) for l in open(os.path.join(D, "runs.jsonl"), encoding="utf-8")]
lines.append(f"\n## runs {len(runs)}")
lines.append("## fixed_tokens runs")
for r in runs:
    if r["strategy"] == "fixed_tokens":
        lines.append(json.dumps({k: r.get(k) for k in ["run_id", "size", "token_budget", "overlap", "chunks", "chunk_words_mean", "chunk_tokens_mean", "chunk_tokens_p95", "metrics"]}, ensure_ascii=False)[:400])
lines.append("\n## dense/rrf runs")
for r in runs:
    if r["retriever"] in ("dense", "rrf"):
        lines.append(json.dumps({k: r.get(k) for k in ["run_id", "retriever", "model", "strategy", "size", "token_budget", "overlap", "chunks", "embedding_sec", "truncation_rate", "metrics"]}, ensure_ascii=False))

lines.append("\n## stage counts")
from collections import Counter
lines.append(str(Counter(r.get("retriever") for r in runs)))
lines.append(str(Counter(r.get("strategy") for r in runs)))

lines.append("\n## dense top metrics vs bm25 at same config")
def find(retriever, strategy, size, ov, norm="N0", model=None):
    for r in runs:
        if r["retriever"] == retriever and r["strategy"] == strategy and r["size"] == size and r["overlap"] == ov and r["normalization"] == norm and (model is None or r.get("model") == model):
            return r
    return None
for retr in ["bm25_ws", "dense", "rrf"]:
    rr = find(retr, "fixed_words", 100, 0.0, model="BAAI/bge-m3" if retr != "bm25_ws" else None)
    if rr:
        lines.append(f"{retr}: {json.dumps(rr['metrics'], ensure_ascii=False)}")

open("results/analysis/dryrun_details.txt", "w", encoding="utf-8").write("\n".join(map(str, lines)) + "\n")
print("ok")
