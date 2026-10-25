import json
from collections import defaultdict

rows = [json.loads(l) for l in open("data/external/urdu_chunking_full/run_index.jsonl", encoding="utf-8")]
dense = [r for r in rows if r["retriever"] == "dense"]
by = defaultdict(lambda: [0.0, 0])
for r in dense:
    by[r["model"]][0] += r.get("embedding_sec", 0) or 0
    by[r["model"]][1] += 1
print("dense runs", len(dense), "total_embedding_hours", round(sum(r.get("embedding_sec", 0) or 0 for r in dense) / 3600, 2))
for m, (s, n) in by.items():
    print(m, n, "runs", round(s / 60, 1), "min")
print("chunk counts by size (fixed_words overlap 0):")
for r in rows:
    if r["strategy"] == "fixed_words" and r["overlap"] == 0.0 and r["normalization"] == "N0" and r["retriever"] == "dense" and r.get("model") == "BAAI/bge-m3":
        print(r["size"], r["chunks"], "questions", r["metrics"]["questions"])
