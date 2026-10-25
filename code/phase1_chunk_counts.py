import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

POOL = json.load(open("data/processed/pool_articles.json", encoding="utf-8"))
TITLES = [a["title"] for a in POOL["articles"]]
FERT = json.load(open("results/phase1/fertility.json", encoding="utf-8"))["summary"]

squad = json.load(open("data/raw/uqa/squad_train.json", encoding="utf-8"))
id2key, order = {}, {}
for art in squad["data"]:
    t = art["title"]
    order[t] = []
    for pi, p in enumerate(art["paragraphs"]):
        order[t].append((t, pi))
        for qa in p["qas"]:
            id2key[qa["id"]] = (t, pi)

rows = pq.read_table("data/raw/uqa/uqa_train.parquet", columns=["id", "context"]).to_pylist()
groups = defaultdict(list)
for r in rows:
    k = id2key.get(r["id"])
    if k is not None:
        groups[k].append(r["context"])
chosen = {}
for k, ctxs in groups.items():
    c = Counter(ctxs)
    chosen[k] = max(c.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]

docs = {t: "\n\n".join(chosen[k] for k in order[t] if k in chosen).split() for t in TITLES}
total_words = sum(len(d) for d in docs.values())


def n_fixed(size, overlap):
    step = max(1, int(size * (1 - overlap)))
    n = 0
    for d in docs.values():
        pos = 0
        while pos < len(d):
            n += 1
            if pos + size >= len(d):
                break
            pos += step
    return n


def n_sent(size, overlap):
    step = max(1, int(size * (1 - overlap)))
    n = 0
    for d in docs.values():
        sents, cur = [], 0
        for w in d:
            cur += 1
            if w.endswith(("\u06d4", "\u061f", "!")):
                sents.append(cur)
                cur = 0
        if cur:
            sents.append(cur)
        i = 0
        while i < len(sents):
            used, j = 0, i
            while j < len(sents) and (used == 0 or used + sents[j] <= size):
                used += sents[j]
                j += 1
            n += 1
            adv, acc = 0, 0
            while i + adv < j and acc < step:
                acc += sents[i + adv]
                adv += 1
            i += max(1, adv)
            if i >= len(sents):
                break
    return n


SIZES = [50, 100, 200, 400, 800]
OVS = [0.0, 0.10, 0.25]
fixed = {s: {o: n_fixed(s, o) for o in OVS} for s in SIZES}
sent = {s: {o: n_sent(s, o) for o in OVS} for s in SIZES}
print("counts done")

lines = ["# Chunk counts for the pre-registered grid (measured, 150-article pool)", "",
         f"Pool: {len(TITLES)} articles, {total_words:,} whitespace words.", "",
         "All counts are exact for the stated rule. Token-defined counts are converted to their",
         "measured word equivalent (tokens / fertility) and then counted with the word rule; Phase 2",
         "replaces this with a real tokenizer-boundary chunker and records the difference.", "",
         "## Fixed-word chunks", "", "| size | overlap 0% | overlap 10% | overlap 25% |", "| --- | --- | --- | --- |"]
for s in SIZES:
    lines.append(f"| {s} | " + " | ".join(f"{fixed[s][o]:,}" for o in OVS) + " |")

lines += ["", "## Sentence-aware chunks (split at ۔ ؟ !)", "",
          "| target words | overlap 0% | overlap 10% | overlap 25% |", "| --- | --- | --- | --- |"]
for s in SIZES:
    lines.append(f"| {s} | " + " | ".join(f"{sent[s][o]:,}" for o in OVS) + " |")

lines += ["", "## Fixed-token chunks, word-equivalent and count (no overlap)", "",
          "| model | tokens/word | 64 tok | 128 tok | 256 tok | 512 tok |", "| --- | --- | --- | --- | --- | --- |"]
MODELS = ["bge-m3", "e5-large-instruct", "qwen3-0.6b"]
for m in MODELS:
    tpw = FERT[m]["ur_tok_per_word"]
    cells = []
    for b in [64, 128, 256, 512]:
        w = max(1, round(b / tpw))
        cells.append(f"{w}w / {fixed[w][0.0] if w in fixed else n_fixed(w, 0.0):,}")
    lines.append(f"| {m} (max {FERT[m]['max_seq']}) | {tpw:.3f} | " + " | ".join(cells) + " |")

primary = turn = 0
lines += ["", "## Passages per model, primary grid (word sizes 50/100/200; overlaps 0/10/25; 5 strategies)", "",
          "| model | passages |", "| --- | --- |"]
totals = {}
for m in MODELS:
    total = sum(4 * fixed[s][o] + sent[s][o] for s in [50, 100, 200] for o in OVS)
    totals[m] = total
    lines.append(f"| {m} | {total:,} |")

lines += ["", "## Extended grid (400/800 words), long-context models only", ""]
for m in ["bge-m3", "qwen3-0.6b"]:
    total = sum(4 * fixed[s][o] + sent[s][o] for s in [400, 800] for o in OVS)
    lines.append(f"- {m}: {total:,} passages")

json.dump({"fixed": {str(k): {str(o): v for o, v in d.items()} for k, d in fixed.items()},
           "sentence": {str(k): {str(o): v for o, v in d.items()} for k, d in sent.items()},
           "primary_totals": totals, "total_words": total_words},
          open("results/phase1/chunk_counts.json", "w", encoding="utf-8"), indent=2)
open("results/phase1/chunk_counts.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
