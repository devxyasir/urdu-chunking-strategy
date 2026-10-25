import json
import os
import re
import sys

PAPERS = json.load(open("configs/phase0_papers.json", encoding="utf-8"))
BY_CLUSTER = {}
for p in PAPERS:
    BY_CLUSTER.setdefault(p["cluster"], []).append(p["key"])

KEYS = [
    "chunk", "size", "overlap", "boundary", "sentence", "token", "recall", "mrr", "ndcg",
    "bm25", "dense", "embedding", "urdu", "multilingual", "normaliz", "low-resource",
]


def clean(t):
    t = t.replace("\u00ad", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"\s+", " ", t)
    return t


def sentences(t):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t)


def digest(key, max_abs=1500, max_hits=16, max_concl=1300):
    path = os.path.join("data", "raw", "papers", key + ".txt")
    t = clean(open(path, encoding="utf-8").read())
    out = ["=" * 90, "### " + key, "HEAD: " + t[:max_abs]]
    hits = []
    for s in sentences(t):
        low = s.lower()
        if len(s) < 60 or len(s) > 420:
            continue
        score = sum(low.count(k) for k in KEYS if k in low)
        if score >= 3 and ("chunk" in low or "recall" in low or "urdu" in low or "bm25" in low):
            hits.append((score, s))
    seen = set()
    printed = 0
    for score, s in sorted(hits, key=lambda x: -x[0]):
        sig = s[:90]
        if sig in seen:
            continue
        seen.add(sig)
        out.append("  * " + s[:380])
        printed += 1
        if printed >= max_hits:
            break
    m = re.search(r"\b(Conclusion|Conclusions|Discussion)\b", t, re.I)
    if m:
        out.append("CONCL: " + t[m.start():m.start() + max_concl])
    return "\n".join(out)


targets = sys.argv[1:] or ["all"]
os.makedirs("results/phase0/digests", exist_ok=True)
for target in targets:
    if target == "all":
        keys = [p["key"] for p in PAPERS]
    elif target in BY_CLUSTER:
        keys = BY_CLUSTER[target]
    else:
        keys = [target]
    body = "\n\n".join(digest(k) for k in keys)
    path = f"results/phase0/digests/{target}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    print("WROTE", path, len(body), "chars")
