import csv
import json
import os

import numpy as np

D = "data/external/urdu_chunking_full"
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = [q for q in csv.DictReader(f)]
TEST = [q for q in QS if q["split"] == "test"]
ART = np.array([int(q["article_id"]) for q in TEST])
ARTICLES = sorted(set(ART.tolist()))
IDX = {a: np.flatnonzero(ART == a) for a in ARTICLES}
RUNS = {}
for line in open(os.path.join(D, "runs.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["split"] == "test" and r["corpus"] == "uqa_ur":
        RUNS[r["run_id"]] = r


def get(size):
    for r in RUNS.values():
        if (r["retriever"] == "bm25_ws" and r["strategy"] == "fixed_words" and r["size"] == size
                and r["overlap"] == 0.0 and r["normalization"] == "N0"):
            return r
    return None


def means(r):
    a = np.asarray(r["per_question"]["bm500x"], float)
    return {art: float(a[IDX[art]].mean()) for art in ARTICLES if len(IDX[art])}


def paired(x, y):
    common = sorted(set(x) & set(y))
    a = np.array([x[k] for k in common])
    b = np.array([y[k] for k in common])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(common), (10000, len(common)))
    d = (a[idx] - b[idx]).mean(1)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float((a - b).mean() * 100), float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100), float(p)


a, b = get(200), get(400)
print("BM25 200-400 pts", paired(means(a), means(b)))
