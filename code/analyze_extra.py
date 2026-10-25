import csv
import json
import os
from collections import defaultdict

import numpy as np

D = "data/external/urdu_chunking_full"
OUT = "results/analysis"
B = 10000
SEED = 0

with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = [q for q in csv.DictReader(f)]
TEST = [q for q in QS if q["split"] == "test"]
ART = np.array([int(q["article_id"]) for q in TEST])
ARTICLES = sorted(set(ART.tolist()))
ART_IDX = {a: np.flatnonzero(ART == a) for a in ARTICLES}

RUNS = {}
with open(os.path.join(D, "runs.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["split"] != "test":
            continue
        pq = r["per_question"]
        if not pq or len(pq["bm500x"]) != len(TEST):
            continue
        RUNS[r["run_id"]] = {"meta": {k: r[k] for k in ["corpus", "retriever", "model", "strategy", "size", "overlap", "token_budget", "normalization"]},
                             "bm500x": np.asarray(pq["bm500x"], np.float64),
                             "bm500": np.asarray(pq["bm500"], np.float64),
                             "split": np.asarray(pq["split"], np.float64)}
print("runs", len(RUNS))


def find(corpus="uqa_ur", retriever="dense", model=None, strategy="fixed_words", size=None, overlap=0.0, norm="N0", budget=None):
    hits = []
    for r in RUNS.values():
        m = r["meta"]
        if m["corpus"] != corpus or m["retriever"] != retriever or m["strategy"] != strategy:
            continue
        if model is not None and m["model"] != model:
            continue
        if size is not None and m["size"] != size:
            continue
        if budget is not None and m["token_budget"] != budget:
            continue
        if overlap is not None and m["overlap"] != overlap:
            continue
        if norm is not None and m["normalization"] != norm:
            continue
        hits.append(r)
    return hits[0] if hits else None


def means(run, metric="bm500x"):
    arr = run[metric]
    out = {}
    for a in ARTICLES:
        v = arr[ART_IDX[a]]
        v = v[v >= 0]
        if len(v):
            out[a] = float(v.mean())
    return out


def paired(a, b):
    common = sorted(set(a) & set(b))
    x = np.array([a[k] for k in common])
    y = np.array([b[k] for k in common])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    d = (x[idx] - y[idx]).mean(1)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float((x - y).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p, len(common)


rows_w, rows_t, rows_o = [], [], []
DENSE = [("bge-m3", "BAAI/bge-m3"), ("e5", "intfloat/multilingual-e5-large-instruct"),
         ("qwen", "Qwen/Qwen3-Embedding-0.6B"), ("labse", "sentence-transformers/LaBSE")]
for label, model in DENSE:
    for a, b in [(50, 100), (100, 200)]:
        ra, rb = find(model=model, size=a), find(model=model, size=b)
        if ra and rb:
            d, lo, hi, p, n = paired(means(ra), means(rb))
            rows_w.append({"model": label, "contrast": f"{a} - {b}", "diff": d, "lo": lo, "hi": hi, "p": p, "n": n})
    budgets = [64, 128, 256]
    for a, b in [(64, 128), (128, 256), (64, 256)]:
        ra, rb = find(model=model, strategy="fixed_tokens", budget=a, size=None), find(model=model, strategy="fixed_tokens", budget=b, size=None)
        if ra and rb:
            d, lo, hi, p, n = paired(means(ra), means(rb))
            rows_t.append({"model": label, "contrast": f"{a}t - {b}t", "diff": d, "lo": lo, "hi": hi, "p": p, "n": n})
    for size in [50, 100, 200]:
        r0 = find(model=model, size=size, overlap=0.0)
        r25 = find(model=model, size=size, overlap=0.25)
        if r0 and r25:
            d, lo, hi, p, n = paired(means(r25), means(r0))
            rows_o.append({"model": label, "size": size, "contrast": "25% - 0%", "diff": d, "lo": lo, "hi": hi, "p": p, "n": n})

for label, model in [("BM25-ws", None)]:
    for size in [50, 100, 200]:
        r0 = find(retriever="bm25_ws", model=None, size=size, overlap=0.0)
        r10 = find(retriever="bm25_ws", model=None, size=size, overlap=0.10)
        r25 = find(retriever="bm25_ws", model=None, size=size, overlap=0.25)
        for tag, r in [("10% - 0%", r10), ("25% - 0%", r25)]:
            if r0 and r:
                d, lo, hi, p, n = paired(means(r), means(r0))
                rows_o.append({"model": label, "size": size, "contrast": tag, "diff": d, "lo": lo, "hi": hi, "p": p, "n": n})

for name, rows in [("word_diffs", rows_w), ("token_diffs", rows_t), ("overlap_diffs", rows_o)]:
    with open(os.path.join(OUT, f"table_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
print("word diffs", len(rows_w), "token diffs", len(rows_t), "overlap diffs", len(rows_o))

# word extent at the best token budget
fert = json.load(open(os.path.join(D, "tokenizer_fertility.json"), encoding="utf-8"))
best_tok = {"bge": 128, "e5": 64, "qwen": 256}
ext = {}
for k, bud in best_tok.items():
    ext[k] = {"best_token_budget": bud, "tokens_per_word": fert[k]["ur"], "implied_words": round(bud / fert[k]["ur"], 1)}
json.dump(ext, open(os.path.join(OUT, "token_budget_word_extent.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(ext, indent=1))
