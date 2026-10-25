"""Run the missing dense lower-size grid (10/25/40/50 words).

This is the decisive experiment for the paper's boundary claim. It reuses the archived
corpus and questions and writes a self-contained result package. It does not modify the
original 360-run package.

Usage:
    python code/run_small_sizes.py --data-dir data/external/urdu_chunking_full \
        --out-dir results/small_sizes --models bge e5 qwen labse --sizes 10 25 40 50 --device cuda
    python code/run_small_sizes.py --models bge --sizes 10 --device cpu   # smoke test only

Requires sentence-transformers and a GPU for the full grid. The script is resumable: it
appends one JSON record per completed (model, size) and skips run IDs already present in
<out-dir>/runs.jsonl. Do not publish the data package; it contains UQA text.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunking import chunk_fixed_words
from metrics import bm_recall, is_relevant_doc, mrr_at_10, ndcg_at_10, recall_at_k

DEFAULT_DATA = "data/external/urdu_chunking_full"
DEFAULT_OUT = "results/small_sizes"
MODELS = {
    "bge": dict(hf="BAAI/bge-m3", max_seq=8192, prefix=None, batch=64),
    "e5": dict(hf="intfloat/multilingual-e5-large-instruct", max_seq=512,
               prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q}", batch=64),
    "qwen": dict(hf="Qwen/Qwen3-Embedding-0.6B", max_seq=32768,
                 prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q}", batch=64),
    "labse": dict(hf="sentence-transformers/LaBSE", max_seq=256, prefix=None, batch=64),
}


def load_data(data_dir):
    docs = {}
    with open(os.path.join(data_dir, "corpus_ur.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                docs[int(r["article_id"])] = r["doc"]
    qs = []
    with open(os.path.join(data_dir, "questions.csv"), encoding="utf-8-sig", newline="") as f:
        for q in csv.DictReader(f):
            if q["split"] != "test":
                continue
            qs.append(dict(qid=q["qid"], doc=int(q["article_id"]), q=q["question"],
                           s=int(q["answer_start"]), e=int(q["answer_end"])))
    return docs, qs


def build_chunks(docs, size):
    cd, cs, ce, cw = [], [], [], []
    for aid in sorted(docs):
        text = docs[aid]
        for s, e in chunk_fixed_words(text, size, 0.0):
            cd.append(aid)
            cs.append(s)
            ce.append(e)
            cw.append(len(text[s:e].split()))
    return np.asarray(cd, np.int32), np.asarray(cs, np.int64), np.asarray(ce, np.int64), np.asarray(cw, np.int64)


def article_means(values, docs):
    by = defaultdict(list)
    for v, a in zip(values, docs):
        by[a].append(v)
    return np.array([np.mean(by[a]) for a in sorted(by)], dtype=np.float64)


def bootstrap_ci(x, b=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(b, dtype=np.float64)
    for i in range(b):
        means[i] = x[rng.integers(0, n, n)].mean()
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def run_id(mk, size):
    return f"sm_test_{mk}_fixed_words_{size}_0_N0_dense"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bge", "e5", "qwen", "labse"], choices=list(MODELS))
    ap.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 40, 50])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    os.makedirs(args.out_dir, exist_ok=True)
    docs, qs = load_data(args.data_dir)
    qdoc = np.array([q["doc"] for q in qs], np.int32)
    qs_span = np.array([[q["s"], q["e"]] for q in qs], np.int64)
    with open(os.path.join(args.out_dir, "question_ids.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "article_id"])
        for q in qs:
            w.writerow([q["qid"], q["doc"]])
    print(f"docs={len(docs)} questions={len(qs)}", flush=True)

    runs_path = os.path.join(args.out_dir, "runs.jsonl")
    done = set()
    if os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["run_id"])
                except Exception:
                    pass
    mode = "a" if done else "w"
    out = open(runs_path, mode, encoding="utf-8")

    try:
        for mk in args.models:
            spec = MODELS[mk]
            todo = [s for s in args.sizes if run_id(mk, s) not in done]
            if not todo:
                print(f"{mk}: all requested sizes already present", flush=True)
                continue
            t0 = time.time()
            model = SentenceTransformer(spec["hf"], device=args.device)
            model.max_seq_length = spec["max_seq"]
            qt = [spec["prefix"].replace("{q}", q["q"]) if spec["prefix"] else q["q"] for q in qs]
            qemb = model.encode(qt, batch_size=args.batch, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True)
            print(f"{mk}: queries encoded in {time.time()-t0:.1f}s", flush=True)
            for size in todo:
                t1 = time.time()
                cd, cs, ce, cw = build_chunks(docs, size)
                texts = [docs[a][s:e] for a, s, e in zip(cd, cs, ce)]
                emb = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=True)
                scores = qemb @ emb.T
                k = min(100, scores.shape[1])
                top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
                top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), 1), 1)
                by_doc = defaultdict(list)
                for j, a in enumerate(cd):
                    by_doc[int(a)].append(j)
                bm500, split = np.zeros(len(qs)), np.zeros(len(qs))
                r1, r3, r5, r10, mrr, ndcg = (np.zeros(len(qs)) for _ in range(6))
                for i in range(len(qs)):
                    any_rel = any(is_relevant_doc(int(cd[j]), (cs[j], ce[j]), int(qdoc[i]), qs_span[i])
                                  for j in by_doc[int(qdoc[i])])
                    ranked = top[i]
                    flags = np.array([is_relevant_doc(int(cd[j]), (cs[j], ce[j]), int(qdoc[i]), qs_span[i])
                                      for j in ranked])
                    hit, _ = bm_recall(ranked, list(zip(cs, ce)), cw, qs_span[i], budget=500,
                                       chunk_docs=cd, qdoc=int(qdoc[i]))
                    bm500[i] = float(hit)
                    split[i] = float(not any_rel)
                    r1[i], r3[i], r5[i], r10[i] = (recall_at_k(flags, n) for n in (1, 3, 5, 10))
                    mrr[i], ndcg[i] = mrr_at_10(flags), ndcg_at_10(flags)
                am = article_means(bm500, qdoc)
                mean, lo, hi = bootstrap_ci(am)
                rec = dict(run_id=run_id(mk, size), model=spec["hf"],
                           strategy="fixed_words", size=size, overlap=0.0, normalization="N0",
                           chunks=int(len(cd)), chunk_words_mean=float(cw.mean()),
                           metrics=dict(bm500x=mean, bm500x_lo=lo, bm500x_hi=hi,
                                        bm500=float(bm500.mean()), answer_split_rate=float(split.mean()),
                                        r1=float(r1.mean()), r3=float(r3.mean()), r5=float(r5.mean()),
                                        r10=float(r10.mean()), mrr10=float(mrr.mean()), ndcg10=float(ndcg.mean())),
                           per_question=dict(bm500x=bm500.tolist(), split=split.tolist()))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done.add(rec["run_id"])
                print(json.dumps({k: rec[k] for k in ("run_id", "chunks", "metrics")}, ensure_ascii=False), flush=True)
                print(f"  size={size} in {time.time()-t1:.1f}s", flush=True)
            del model, qemb
            if args.device == "cuda":
                import torch
                torch.cuda.empty_cache()
    finally:
        out.close()

    records = []
    with open(runs_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    summary = [dict(run_id=r["run_id"], model=r["model"], size=r["size"],
                    chunks=r["chunks"], bm500x=r["metrics"]["bm500x"],
                    lo=r["metrics"]["bm500x_lo"], hi=r["metrics"]["bm500x_hi"]) for r in records]
    summary.sort(key=lambda r: (r["model"], r["size"]))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("wrote", runs_path)


if __name__ == "__main__":
    main()
