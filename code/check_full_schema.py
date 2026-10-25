import csv
import json
import os
from collections import Counter

D = "data/external/urdu_chunking_full"
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))
test = [q for q in qs if q["split"] == "test"]
dev = [q for q in qs if q["split"] == "dev"]
print("test", len(test), "dev", len(dev))
print("test articles", len({q["article_id"] for q in test}), "dev articles", len({q["article_id"] for q in dev}))
print("article id sample", test[0]["article_id"], "qid", test[0]["qid"])
print("article min/max", min(int(q["article_id"]) for q in test), max(int(q["article_id"]) for q in test))

first = None
n = 0
with open(os.path.join(D, "runs.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        n += 1
        if r["retriever"] == "dense" and r["corpus"] == "uqa_ur" and r.get("model") == "BAAI/bge-m3" and r["strategy"] == "fixed_words" and r["size"] == 50 and r["overlap"] == 0.0:
            first = r
            break
print("scanned", n)
print("keys", list(first.keys()))
pq = first["per_question"]
print("pq keys", list(pq.keys()))
for k, v in pq.items():
    print(k, len(v), "neg1", sum(1 for x in v if x == -1))
print("metrics", first["metrics"])
