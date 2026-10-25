import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

D = "data/external/urdu_chunking_full"
OUT = "results/analysis"
os.makedirs(OUT, exist_ok=True)
B = 10000
SEED = 0

with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = [q for q in csv.DictReader(f)]
TEST = [q for q in QS if q["split"] == "test"]
QID = [q["qid"] for q in TEST]
ART = np.array([int(q["article_id"]) for q in TEST])
ARTICLES = sorted(set(ART.tolist()))
ART_IDX = {a: np.flatnonzero(ART == a) for a in ARTICLES}

METRICS = ["bm500x", "bm500", "r10", "mrr10", "split", "words_used", "bm200x", "bm1000x", "bm2000x"]
RUNS = {}
manifest = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))

with open(os.path.join(D, "runs.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["split"] != "test" and r["corpus"] != "native_ur":
            continue
        pq = r["per_question"]
        if not pq:
            continue
        n = len(next(iter(pq.values())))
        if n != len(QID):
            continue
        RUNS[r["run_id"]] = {
            "meta": {k: r[k] for k in ["run_id", "corpus", "split", "retriever", "model", "model_revision",
                                       "strategy", "size", "overlap", "token_budget", "normalization",
                                       "chunks", "truncated", "truncation_rate", "embedding_sec", "index_bytes"]},
            "m": {k: np.asarray(pq[k], dtype=np.float64) for k in METRICS if k in pq},
        }
print("loaded runs", len(RUNS))


def key_match(meta, corpus=None, retriever=None, model=None, strategy=None, size=None, overlap=None,
              norm=None, budget=None):
    if corpus is not None and meta["corpus"] != corpus:
        return False
    if retriever is not None and meta["retriever"] != retriever:
        return False
    if model is not None and meta["model"] != model:
        return False
    if strategy is not None and meta["strategy"] != strategy:
        return False
    if size is not None and meta["size"] != size:
        return False
    if overlap is not None and meta["overlap"] != overlap:
        return False
    if norm is not None and meta["normalization"] != norm:
        return False
    if budget is not None and meta["token_budget"] != budget:
        return False
    return True


def find(**kw):
    out = [v for v in RUNS.values() if key_match(v["meta"], **kw)]
    if len(out) > 1:
        # prefer exact single match; if still multiple, return first with warning
        ids = [o["meta"]["run_id"] for o in out]
        return out[0]
    return out[0] if out else None


def article_means(run, metric="bm500x", valid_mask=None):
    arr = run["m"][metric]
    out = {}
    for a in ARTICLES:
        idx = ART_IDX[a]
        if valid_mask is not None:
            idx = idx[valid_mask[idx]]
        vals = arr[idx]
        vals = vals[vals >= 0]
        if len(vals):
            out[a] = float(vals.mean())
    return out


def ci(means, seed=SEED):
    v = np.array([means[a] for a in sorted(means)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    boot = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(v)


def paired(a, b, seed=SEED):
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    x = np.array([a[k] for k in common])
    y = np.array([b[k] for k in common])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    d = (x[idx] - y[idx]).mean(axis=1)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float((x - y).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p, len(common)


def paired_question_means(run_a, run_b, metric="bm500x"):
    a = run_a["m"][metric]
    b = run_b["m"][metric]
    mask = (a >= 0) & (b >= 0)
    ma, mb = {}, {}
    for art in ARTICLES:
        idx = ART_IDX[art]
        idx = idx[mask[idx]]
        if len(idx):
            ma[art] = float(a[idx].mean())
            mb[art] = float(b[idx].mean())
    return ma, mb


# ---------- tables ----------
words_rows = []
models = [("BM25-ws", "bm25_ws", None), ("BM25-char", "bm25_char", None),
          ("bge-m3", "dense", "BAAI/bge-m3"),
          ("e5-large-instruct", "dense", "intfloat/multilingual-e5-large-instruct"),
          ("Qwen3-0.6B", "dense", "Qwen/Qwen3-Embedding-0.6B"),
          ("LaBSE", "dense", "sentence-transformers/LaBSE")]

for label, retr, model in models:
    for size in [50, 100, 200, 400, 800]:
        r = find(corpus="uqa_ur", retriever=retr, model=model, strategy="fixed_words",
                 size=size, overlap=0.0, norm="N0")
        if r is None:
            continue
        m, lo, hi, n = ci(article_means(r))
        words_rows.append({"model": label, "retriever": retr, "size": size, "metric": "bm500x",
                           "mean": m, "lo": lo, "hi": hi, "n_articles": n, "chunks": r["meta"]["chunks"]})

token_rows = []
for label, retr, model in models:
    budgets = [64, 128, 256] if retr == "dense" else [50, 100, 200, 400, 800]
    for bud in budgets:
        r = find(corpus="uqa_ur", retriever=retr, model=model, strategy="fixed_tokens",
                 budget=bud, overlap=0.0, norm="N0")
        if r is None:
            continue
        m, lo, hi, n = ci(article_means(r))
        token_rows.append({"model": label, "retriever": retr, "token_budget": bud, "metric": "bm500x",
                           "mean": m, "lo": lo, "hi": hi, "n_articles": n, "chunks": r["meta"]["chunks"]})

strategy_rows = []
diff_rows = []
for label, retr, model in models:
    for size in [50, 100, 200]:
        base = find(corpus="uqa_ur", retriever=retr, model=model, strategy="fixed_words",
                    size=size, overlap=0.0, norm="N0")
        for strat in ["fixed_words", "sentence_aware", "recursive_english", "recursive_urdu"]:
            r = find(corpus="uqa_ur", retriever=retr, model=model, strategy=strat, size=size,
                     overlap=0.0, norm="N0")
            if r is None:
                continue
            m, lo, hi, n = ci(article_means(r))
            strategy_rows.append({"model": label, "retriever": retr, "strategy": strat, "size": size,
                                  "mean": m, "lo": lo, "hi": hi, "n_articles": n})
        if base is not None:
            for strat in ["sentence_aware", "recursive_urdu", "recursive_english"]:
                r = find(corpus="uqa_ur", retriever=retr, model=model, strategy=strat, size=size,
                         overlap=0.0, norm="N0")
                if r is None:
                    continue
                res = paired(article_means(r), article_means(base))
                if res:
                    diff_rows.append({"model": label, "size": size, "contrast": f"{strat} - fixed_words",
                                      "diff": res[0], "lo": res[1], "hi": res[2], "p": res[3], "n_articles": res[4]})
            for a, b in [("sentence_aware", "recursive_english"), ("recursive_urdu", "recursive_english"),
                         ("sentence_aware", "recursive_urdu")]:
                ra = find(corpus="uqa_ur", retriever=retr, model=model, strategy=a, size=size, overlap=0.0, norm="N0")
                rb = find(corpus="uqa_ur", retriever=retr, model=model, strategy=b, size=size, overlap=0.0, norm="N0")
                if ra is None or rb is None:
                    continue
                res = paired(article_means(ra), article_means(rb))
                if res:
                    diff_rows.append({"model": label, "size": size, "contrast": f"{a} - {b}",
                                      "diff": res[0], "lo": res[1], "hi": res[2], "p": res[3], "n_articles": res[4]})

overlap_rows = []
for label, retr, model in models:
    for size in [50, 100, 200]:
        for ov in [0.0, 0.10, 0.25]:
            r = find(corpus="uqa_ur", retriever=retr, model=model, strategy="fixed_words",
                     size=size, overlap=ov, norm="N0")
            if r is None:
                continue
            m, lo, hi, n = ci(article_means(r))
            overlap_rows.append({"model": label, "retriever": retr, "size": size, "overlap": ov,
                                 "mean": m, "lo": lo, "hi": hi, "n_articles": n})

retr_rows = []
for size in [50, 100, 200]:
    for strat in ["fixed_words", "sentence_aware"]:
        combos = []
        for label, retr, model in models:
            r = find(corpus="uqa_ur", retriever=retr, model=model, strategy=strat, size=size,
                     overlap=0.0, norm="N0")
            if r is None:
                continue
            m, lo, hi, n = ci(article_means(r))
            combos.append((label, m, lo, hi))
            if retr == "dense":
                rrf = find(corpus="uqa_ur", retriever="rrf", model=model, strategy=strat, size=size,
                           overlap=0.0, norm="N0")
                if rrf is not None:
                    m2, lo2, hi2, _ = ci(article_means(rrf))
                    combos.append((label + " + RRF", m2, lo2, hi2))
        for label, m, lo, hi in combos:
            retr_rows.append({"size": size, "strategy": strat, "system": label, "mean": m, "lo": lo, "hi": hi})

norm_rows = []
norm_diff_rows = []
for retr in ["bm25_ws", "bm25_char"]:
    for size in [50, 100, 200]:
        runs = {nz: find(corpus="uqa_ur", retriever=retr, model=None, strategy="fixed_words",
                         size=size, overlap=0.0, norm=nz) for nz in ["N0", "N1", "N2"]}
        for nz, r in runs.items():
            if r is None:
                continue
            m, lo, hi, n = ci(article_means(r))
            norm_rows.append({"retriever": retr, "size": size, "normalization": nz,
                              "mean": m, "lo": lo, "hi": hi, "n_articles": n})
        for nz in ["N1", "N2"]:
            if runs["N0"] is None or runs[nz] is None:
                continue
            ma, mb = paired_question_means(runs[nz], runs["N0"])
            res = paired(ma, mb)
            if res:
                norm_diff_rows.append({"retriever": retr, "size": size, "contrast": f"{nz} - N0",
                                       "diff": res[0], "lo": res[1], "hi": res[2], "p": res[3], "n_articles": res[4]})

eng_rows = []
eng_diff_rows = []
for strat in ["fixed_words", "recursive_english", "sentence_aware", "recursive_urdu"]:
    for size in [50, 100, 200]:
        ru = find(corpus="uqa_ur", retriever="dense", model="BAAI/bge-m3", strategy=strat, size=size, overlap=0.0, norm="N0")
        en = find(corpus="squad_en", retriever="dense", model="BAAI/bge-m3", strategy=strat, size=size, overlap=0.0, norm="N0")
        if ru is None or en is None:
            continue
        for tag, r in [("Urdu", ru), ("English", en)]:
            m, lo, hi, n = ci(article_means(r))
            eng_rows.append({"language": tag, "strategy": strat, "size": size, "mean": m, "lo": lo, "hi": hi, "n_articles": n})
        res = paired(article_means(en), article_means(ru))
        if res:
            eng_diff_rows.append({"strategy": strat, "size": size, "contrast": "English - Urdu",
                                  "diff": res[0], "lo": res[1], "hi": res[2], "p": res[3], "n_articles": res[4]})

# ---------- headline summaries ----------
def argmax_table(rows, unit):
    out = {}
    for label in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == label]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["mean"])
        key = best["size"] if unit == "words" else best["token_budget"]
        out[label] = {"argmax": key, "mean": best["mean"], "lo": best["lo"], "hi": best["hi"],
                      "curve": {str(r["size"] if unit == "words" else r["token_budget"]): r["mean"] for r in sub}}
    return out


DENSE_LABELS = ["bge-m3", "e5-large-instruct", "Qwen3-0.6B", "LaBSE"]
word_rows_main = [r for r in words_rows if r["model"] in DENSE_LABELS and r["size"] in (50, 100, 200)]
token_rows_main = [r for r in token_rows if r["model"] in DENSE_LABELS and r["token_budget"] in (64, 128, 256)]
word_arg = argmax_table(word_rows_main, "words")
token_arg = argmax_table(token_rows_main, "tokens")
word_arg_ext = argmax_table([r for r in words_rows if r["model"] in ("bge-m3", "Qwen3-0.6B")], "words")

word_vals = [v["argmax"] for v in word_arg.values() if v["argmax"] in (50, 100, 200)]
token_vals = [v["argmax"] for v in token_arg.values() if v["argmax"] in (64, 128, 256)]

head = {"word_argmax": word_arg, "token_argmax": token_arg, "word_argmax_extended": word_arg_ext,
        "word_agreement": len(set(word_vals)), "n_models_word": len(word_vals),
        "token_agreement": len(set(token_vals)), "n_models_token": len(token_vals),
        "self_test_mapping": manifest.get("self_tests", {}).get("4 offset mapping N0/N1/N2"),
        "prefix_checks": manifest.get("prefix_checks")}

for name, rows in [("words", words_rows), ("tokens", token_rows), ("strategies", strategy_rows),
                   ("strategy_diffs", diff_rows), ("overlap", overlap_rows), ("retrievers", retr_rows),
                   ("normalization", norm_rows), ("normalization_diffs", norm_diff_rows),
                   ("english", eng_rows), ("english_diffs", eng_diff_rows)]:
    path = os.path.join(OUT, f"table_{name}.csv")
    if not rows:
        continue
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

json.dump(head, open(os.path.join(OUT, "headline.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(head, indent=1)[:4000])
print("runs used", len(RUNS))
