import csv
import json

D = "data/external/urdu_chunking_full"
rows = []
with open(f"{D}/run_index.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["corpus"] != "uqa_ur" or r["split"] != "test":
            continue
        rows.append(r)

lines = ["# Split rates and strict-vs-crossing check", ""]
lines.append("## Answer-split rate (share of questions with no chunk containing the full answer)")
lines.append("| model | strategy | size | split | bm500 | bm500x |")
lines.append("| --- | --- | --- | --- | --- | --- |")
models = [("BM25-ws", "bm25_ws", None), ("bge-m3", "dense", "BAAI/bge-m3"),
          ("e5", "dense", "intfloat/multilingual-e5-large-instruct"),
          ("qwen", "dense", "Qwen/Qwen3-Embedding-0.6B"), ("labse", "dense", "sentence-transformers/LaBSE")]
for label, retr, model in models:
    for strat in ["fixed_words", "sentence_aware"]:
        for size in [50, 100, 200]:
            hit = [r for r in rows if r["retriever"] == retr and r.get("model") == model and r["strategy"] == strat
                   and r["size"] == size and r["overlap"] == 0.0 and r["normalization"] == "N0"]
            if not hit:
                continue
            m = hit[0]["metrics"]
            lines.append(f"| {label} | {strat} | {size} | {m['answer_split_rate']:.4f} | {m['bm500']:.4f} | {m['bm500x']:.4f} |")

lines.append("")
lines.append("## Strict vs crossing argmax (fixed_words, overlap 0)")
for label, retr, model in models:
    sub = [r for r in rows if r["retriever"] == retr and r.get("model") == model and r["strategy"] == "fixed_words"
           and r["overlap"] == 0.0 and r["normalization"] == "N0" and r["size"] in (50, 100, 200, 400, 800)]
    if not sub:
        continue
    best_strict = max(sub, key=lambda r: r["metrics"]["bm500"])
    best_cross = max(sub, key=lambda r: r["metrics"]["bm500x"])
    lines.append(f"- {label}: strict argmax {best_strict['size']} ({best_strict['metrics']['bm500']:.4f}); "
                 f"crossing argmax {best_cross['size']} ({best_cross['metrics']['bm500x']:.4f})")

open("results/analysis/split_and_budget.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
