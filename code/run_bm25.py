import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

import bm25 as bm25mod
import chunking
import metrics
import normalize as nz

OUT = "results/phase4"
os.makedirs(OUT, exist_ok=True)


def load(split, limit=None):
    recs = []
    for line in open("data/processed/corpus_ur.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == split:
            recs.append(r)
    if limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(recs), min(limit, len(recs)), replace=False)
        recs = [recs[i] for i in sorted(idx)]
    return recs


def run_config(recs, strategy, size, overlap, variant="N0", qlimit=0):
    chunk_texts, spans, doc_ids = [], [], []
    qdata = []
    for ai, r in enumerate(recs):
        doc_norm, imap = nz.normalize_with_map(r["doc"], variant)
        spans_raw = chunking.CHUNKERS[strategy](doc_norm, size, overlap) if strategy in chunking.CHUNKERS else []
        base = len(chunk_texts)
        for s, e in spans_raw:
            chunk_texts.append(doc_norm[s:e])
            spans.append((s, e))
            doc_ids.append(ai)
        for q in r["questions"]:
            gs, status = nz.map_gold(doc_norm, q["answer"], q["start"], q["end"], imap, variant)
            if status != "ok":
                continue
            qdata.append({
                "ai": ai, "q": nz.normalize(q["question"], variant), "gold": gs,
                "base": base, "n_chunks": len(spans_raw),
            })
    if qlimit and len(qdata) > qlimit:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(qdata), qlimit, replace=False))
        qdata = [qdata[i] for i in keep]
    index = bm25mod.BM25([bm25mod.tokenize_ws(t) for t in chunk_texts])
    words = [len(t.split()) for t in chunk_texts]
    per_q = []
    for item in qdata:
        ids, _ = index.topk(bm25mod.tokenize_ws(item["q"]), 100)
        ranked = [int(i) for i in ids]
        flags = metrics.relevance_flags(ranked, spans, item["gold"], chunk_docs=doc_ids, qdoc=item["ai"])
        hit, used = metrics.bm_recall(ranked, spans, words, item["gold"], 500, chunk_docs=doc_ids, qdoc=item["ai"])
        per_q.append({
            "ai": item["ai"], "r1": metrics.recall_at_k(flags, 1), "r3": metrics.recall_at_k(flags, 3),
            "r5": metrics.recall_at_k(flags, 5), "r10": metrics.recall_at_k(flags, 10),
            "mrr": metrics.mrr_at_10(flags), "ndcg": metrics.ndcg_at_10(flags),
            "bm500": float(hit), "split": float(not flags.any()), "words": used,
        })
    return per_q, len(chunk_texts)


def agg(per_q):
    out = {}
    for k in ["r1", "r3", "r5", "r10", "mrr", "ndcg", "bm500", "split", "words"]:
        out[k] = float(np.mean([x[k] for x in per_q]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--qlimit", type=int, default=500)
    ap.add_argument("--out", default="bm25_dev.json")
    args = ap.parse_args()
    recs = load(args.split, limit=30)
    configs = []
    for size in [50, 100, 200]:
        for ov in [0.0, 0.10, 0.25]:
            configs.append(("fixed_words", size, ov))
    for size in [50, 100, 200]:
        configs.append(("recursive_urdu", size, 0.0))
        configs.append(("recursive_english", size, 0.0))
        configs.append(("sentence_aware", size, 0.0))
    rows = []
    t0 = time.time()
    for strategy, size, ov in configs:
        pq, nch = run_config(recs, strategy, size, ov, qlimit=args.qlimit)
        a = agg(pq)
        a.update({"strategy": strategy, "size": size, "overlap": ov, "chunks": nch, "questions": len(pq)})
        rows.append(a)
        print(f"{strategy:18s} {size:4d} {ov:.2f} chunks={nch:5d} bm500={a['bm500']:.3f} r10={a['r10']:.3f}")
    rows.sort(key=lambda r: -r["bm500"])
    json.dump({"split": args.split, "articles": len(recs), "rows": rows,
               "seconds": round(time.time() - t0, 1)},
              open(os.path.join(OUT, args.out), "w", encoding="utf-8"), indent=2)
    lines = ["# BM25 pilot — " + args.split, "",
             f"{len(recs)} articles, {args.qlimit} questions per config, variant N0, seed 0.", "",
             "| strategy | size | overlap | chunks | BM-Recall@500 | Recall@10 | MRR@10 | split rate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['strategy']} | {r['size']} | {r['overlap']:.0%} | {r['chunks']} | "
                     f"{r['bm500']:.3f} | {r['r10']:.3f} | {r['mrr']:.3f} | {r['split']:.3f} |")
    lines.append("")
    lines.append(f"Runtime {time.time() - t0:.0f}s.")
    open(os.path.join(OUT, "bm25_pilot.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
