import json
import os

import numpy as np

import bm25 as bm25mod
import chunking
import normalize as nz

OUT = "results/phase2"
os.makedirs(OUT, exist_ok=True)

recs = [json.loads(l) for l in open("data/processed/corpus_ur.jsonl", encoding="utf-8") if json.loads(l)["split"] == "dev"]

cov_lines = ["# Phase 2 coverage check (real documents, dev split)", "",
             "Non-whitespace character coverage and exact overlap for every config on 10 articles.", "",
             "| strategy | size | overlap | articles | uncovered chars | overlap exact |",
             "| --- | --- | --- | --- | --- | --- |"]
bad = 0
for strategy in ["fixed_words", "sentence_aware", "recursive_english", "recursive_urdu"]:
    for size in [50, 100, 200]:
        for ov in [0.0, 0.10, 0.25]:
            unc = 0
            for r in recs[:10]:
                text = nz.normalize(r["doc"], "N0")
                spans = chunking.CHUNKERS[strategy](text, size, ov)
                covered = np.zeros(len(text), dtype=bool)
                for s, e in spans:
                    covered[s:e] = True
                for i, ch in enumerate(text):
                    if not ch.isspace() and not covered[i]:
                        unc += 1
            cov_lines.append(f"| {strategy} | {size} | {ov:.0%} | 10 | {unc} | n/a |")
            bad += unc
cov_lines.append("")
cov_lines.append(f"Total uncovered non-whitespace characters: {bad}")

# 5 examples
texts, spans, docs, base = [], [], [], []
for ai, r in enumerate(recs):
    text = nz.normalize(r["doc"], "N0")
    chunk_spans = chunking.chunk_fixed_words(text, 200, 0.0)
    off = len(texts)
    for s, e in chunk_spans:
        texts.append(text[s:e])
        spans.append((s, e))
        docs.append(ai)
index = bm25mod.BM25([bm25mod.tokenize_ws(t) for t in texts])
words = [len(t.split()) for t in texts]

rng = np.random.default_rng(0)
picks = rng.choice(len(recs), 5, replace=False)
lines = ["# Five full examples (fixed_words 200, overlap 0, BM25)", ""]
for ai in picks:
    r = recs[ai]
    q = r["questions"][int(rng.integers(0, len(r["questions"])))]
    text = nz.normalize(r["doc"], "N0")
    gold = (q["start"], q["end"])
    ids, scores = index.topk(bm25mod.tokenize_ws(q["question"]), 3)
    lines.append(f"## {r['title']}")
    lines.append("")
    lines.append(f"**Question:** {q['question']}")
    lines.append("")
    lines.append(f"**Gold answer:** {q['answer']}  (chars {gold[0]}-{gold[1]})")
    lines.append("")
    gold_chunk = next((i for i, sp in enumerate(spans) if docs[i] == ai and sp[0] <= gold[0] and sp[1] >= gold[1]), None)
    if gold_chunk is not None:
        lines.append(f"**Gold chunk {gold_chunk}:** {texts[gold_chunk][:420]}")
    lines.append("")
    lines.append("**Top 3:**")
    lines.append("")
    for rank, cid in enumerate(ids, 1):
        rel = docs[cid] == ai and spans[cid][0] <= gold[0] and spans[cid][1] >= gold[1]
        lines.append(f"{rank}. relevant={rel} doc={docs[cid]} score={scores[rank-1]:.3f} :: {texts[cid][:300]}")
    lines.append("")

open(os.path.join(OUT, "coverage.md"), "w", encoding="utf-8").write("\n".join(cov_lines) + "\n")
open(os.path.join(OUT, "examples.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("coverage bad", bad)
