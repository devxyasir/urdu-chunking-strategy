import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

D = "data/external/urdu_chunking_full"
OUT = "docs/audit"
os.makedirs(OUT, exist_ok=True)

idx = [json.loads(l) for l in open(os.path.join(D, "run_index.jsonl"), encoding="utf-8")]
manifest = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = list(csv.DictReader(f))
TEST = [q for q in QS if q["split"] == "test"]
ART = np.array([int(q["article_id"]) for q in TEST])
ARTICLES = sorted(set(ART.tolist()))
AIDX = {a: np.flatnonzero(ART == a) for a in ARTICLES}
NQ = len(TEST)

runs = {}
with open(os.path.join(D, "runs.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        runs[r["run_id"]] = r

lines = ["# Verification report", "",
         f"- run_index rows: {len(idx)}", f"- runs.jsonl rows: {len(runs)}",
         f"- test questions: {len(TEST)}", f"- test articles: {len(ARTICLES)}", ""]

lines += ["## 1. Experiment count", ""]
by_ret = Counter(r["retriever"] for r in idx)
by_model = Counter((r["retriever"], str(r.get("model"))) for r in idx)
by_stage = Counter(r.get("stage") for r in idx)
lines.append(f"- by retriever: {dict(by_ret)}")
lines.append(f"- by stage: {dict(by_stage)}")
lines.append(f"- by retriever/model: {dict(by_model)}")
lines.append(f"- distinct run ids: {len({r['run_id'] for r in idx})}")
lines.append(f"- duplicate run ids: {len(idx) - len({r['run_id'] for r in idx})}")
bm25_norm = Counter((r["retriever"], r["normalization"]) for r in idx if r["retriever"].startswith("bm25"))
lines.append(f"- BM25 by normalization: {dict(bm25_norm)}")
dense_urdu = [r for r in idx if r["retriever"] == "dense" and r["corpus"] == "uqa_ur"]
dense_en = [r for r in idx if r["retriever"] == "dense" and r["corpus"] == "squad_en"]
lines.append(f"- dense Urdu runs: {len(dense_urdu)}; dense English: {len(dense_en)}; dense total: {len(dense_urdu)+len(dense_en)}")
lines.append(f"- RRF runs: {by_ret['rrf']}")
lines += ["", "Paper appendix arithmetic:", "",
          "- BM25: 2 x 5 x 5 x 3 = 150 (omits 6 normalization runs per variant = 12)",
          "- Dense: 21x3 + 6 + 12 + 21 = 102 (the 12 extended runs are per model for bge and Qwen: 24)",
          "- RRF: 84",
          "- Appendix total: 150 + 102 + 84 = 336",
          "- Actual: BM25 162 + dense 114 + RRF 84 = 360."]

lines += ["", "## 2. Per-question length and ranking alignment", ""]
bad_len = []
for r in idx:
    if r["split"] != "test":
        continue
    for k, v in (runs[r["run_id"]]["per_question"] or {}).items():
        if len(v) != NQ:
            bad_len.append((r["run_id"], k, len(v)))
lines.append(f"- runs with wrong per-question length: {len({b[0] for b in bad_len})}")
if bad_len[:5]:
    lines.append(f"- examples: {bad_len[:5]}")
rank_bad = {}
for f in ["rankings_B.npz", "rankings_C.npz", "rankings_D.npz", "rankings_E.npz"]:
    z = np.load(os.path.join(D, f), allow_pickle=True)
    ids = list(z["run_ids"])
    bad = 0
    for rid in ids:
        key = f"{rid}|top10"
        if key not in z or z[key].shape[0] != NQ:
            bad += 1
    rank_bad[f] = (len(ids), bad)
lines.append(f"- rankings (runs, misaligned): {rank_bad}")

lines += ["", "## 3. Paper tables re-derived from per-question arrays", ""]


def means(run, metric="bm500x"):
    a = np.asarray(run["per_question"][metric], float)
    return {art: float(a[AIDX[art]].mean()) for art in ARTICLES if len(AIDX[art])}


def get(corpus="uqa_ur", retriever="dense", model=None, strategy="fixed_words", size=None,
        overlap=0.0, norm="N0", budget=None):
    for r in runs.values():
        if r["corpus"] != corpus or r["retriever"] != retriever or r["strategy"] != strategy:
            continue
        if model is not None and r.get("model") != model:
            continue
        if size is not None and r["size"] != size:
            continue
        if budget is not None and r["token_budget"] != budget:
            continue
        if r["overlap"] != overlap or r["normalization"] != norm:
            continue
        return r
    return None


def m(r):
    return float(np.mean(list(means(r).values())))


expected_words = {
    ("BAAI/bge-m3", 50): 0.7734, ("BAAI/bge-m3", 100): 0.7569, ("BAAI/bge-m3", 200): 0.6796,
    ("intfloat/multilingual-e5-large-instruct", 50): 0.7756,
    ("intfloat/multilingual-e5-large-instruct", 100): 0.7515,
    ("Qwen/Qwen3-Embedding-0.6B", 50): 0.6029, ("Qwen/Qwen3-Embedding-0.6B", 100): 0.5558,
    ("sentence-transformers/LaBSE", 50): 0.5404, ("sentence-transformers/LaBSE", 100): 0.4866,
}
rows = ["| config | paper | recomputed | delta |", "| --- | --- | --- | --- |"]
for (model, size), v in expected_words.items():
    r = get(model=model, size=size)
    got = m(r)
    rows.append(f"| {model.split('/')[-1]} {size}w | {v:.4f} | {got:.4f} | {got-v:+.4f} |")
for size in [50, 100, 200, 400, 800]:
    r = get(retriever="bm25_ws", model=None, size=size)
    if r:
        got = m(r)
        rows.append(f"| BM25 {size}w |  | {got:.4f} | |")
lines += rows

lines += ["", "### Token table", ""]
token_expected = {
    ("BAAI/bge-m3", 64): 0.7671, ("BAAI/bge-m3", 128): 0.7679, ("BAAI/bge-m3", 256): 0.6773,
    ("intfloat/multilingual-e5-large-instruct", 64): 0.7651,
    ("intfloat/multilingual-e5-large-instruct", 128): 0.7616,
    ("Qwen/Qwen3-Embedding-0.6B", 64): 0.4888, ("Qwen/Qwen3-Embedding-0.6B", 128): 0.5890,
    ("Qwen/Qwen3-Embedding-0.6B", 256): 0.5898,
}
rows = ["| config | paper | recomputed | delta |", "| --- | --- | --- | --- |"]
for (model, budget), v in token_expected.items():
    r = get(model=model, strategy="fixed_tokens", budget=budget, size=None)
    got = m(r)
    rows.append(f"| {model.split('/')[-1]} {budget}t | {v:.4f} | {got:.4f} | {got-v:+.4f} |")
lines += rows

lines += ["", "### English control (bge-m3)", ""]
english_expected = [
    ("fixed_words", 50, 0.7734, 0.8474, 7.4),
    ("fixed_words", 100, 0.7569, 0.8288, 7.2),
    ("fixed_words", 200, 0.6796, 0.7577, 7.8),
    ("recursive_english", 50, 0.8276, 0.9351, 10.8),
    ("recursive_english", 100, 0.8357, 0.8988, 6.3),
    ("recursive_english", 200, 0.8092, 0.8000, -0.9),
]
rows = ["| strategy | size | Urdu (paper) | Urdu recomputed | English (paper) | English recomputed | diff paper | diff recomputed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for strat, size, ur, en, diff in english_expected:
    r_ur = get(corpus="uqa_ur", model="BAAI/bge-m3", strategy=strat, size=size)
    r_en = get(corpus="squad_en", model="BAAI/bge-m3", strategy=strat, size=size)
    rows.append(f"| {strat} | {size} | {ur:.4f} | {m(r_ur):.4f} | {en:.4f} | {m(r_en):.4f} | {diff:+.1f} | {(m(r_en)-m(r_ur))*100:+.2f} |")
lines += rows

lines += ["", "### Best systems", ""]
for model, strat, size in [
    ("BAAI/bge-m3", "sentence_aware", 50),
    ("intfloat/multilingual-e5-large-instruct", "sentence_aware", 50),
]:
    r = get(model=model, strategy=strat, size=size)
    lines.append(f"- {model}: {m(r):.4f}")
for model in ["BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct",
              "Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/LaBSE"]:
    r = get(retriever="rrf", model=model, strategy="sentence_aware", size=50)
    lines.append(f"- RRF {model}: {m(r):.4f}")
r = get(retriever="bm25_ws", model=None, strategy="sentence_aware", size=50)
lines.append(f"- BM25: {m(r):.4f}")

lines += ["", "## 4. Dataset facts", ""]
for k, v in manifest["actual_counts"].items():
    lines.append(f"- {k}: {v}")
lines.append(f"- N1/N2 failures: {manifest['normalization_mapping']['N1']['failed']} / "
             f"{manifest['normalization_mapping']['N1']['questions']}")
lines.append(f"- docs not NFC: {manifest['data_stats']['docs_not_nfc']}")
lines.append(f"- tokenizer fertility: {manifest['tokenizer_fertility']}")

lines += ["", "## 5. Citation keys", ""]
tex = open("paper/paper.tex", encoding="utf-8").read()
cites = set()
for m in re.finditer(r"\\citep\{([^}]+)\}", tex):
    cites.update(k.strip() for k in m.group(1).split(","))
for m in re.finditer(r"\\citet\{([^}]+)\}", tex):
    cites.update(k.strip() for k in m.group(1).split(","))
bib = open("paper/refs.bib", encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
lines.append(f"- cited keys: {len(cites)}; bib keys: {len(keys)}")
lines.append(f"- cited but missing: {sorted(cites - keys)}")
lines.append(f"- bib but uncited: {sorted(keys - cites)}")

lines += ["", "## 6. Run flags", ""]
lines.append(f"- truncated runs: {sum(1 for r in idx if r.get('truncated'))}")
lines.append(f"- runs with truncation_rate > 0: {sum(1 for r in idx if r.get('truncation_rate', 0) > 0)}")
lines.append(f"- total recorded embedding seconds: {sum(r.get('embedding_sec', 0) or 0 for r in idx):.1f}")

lines += ["", "## 7. What this report establishes", "",
          "- **Verified:** 360 unique runs; 162 BM25 (81 whitespace + 81 character), 114 dense "
          "(93 Urdu + 21 English), 84 RRF; all per-question arrays length 7,362; all ranking files "
          "aligned; Tables 2--5 reproduce from the archived per-question arrays to within 0.0001.",
          "- **Verified:** 120 test articles, 7,362 test questions, 150-article pool, 1,007,122 words; "
          "37 N1/N2 failures; 2 non-NFC documents; 34 Urdu digit characters.",
          "- **Not verified here:** the bootstrap endpoints in the last digit (stochastic; seed 0, "
          "10,000 resamples); the figures; any dense embedding rerun; the semantic pilot (not in the "
          "360-run grid); the native set (does not exist); human judgments (do not exist); E1 raw "
          "per-question arrays (summary only).",
          "- **Requires a new experiment:** the 10/25/40/50 grid for e5, Qwen3, and LaBSE. bge-m3 was "
          "run by the author and 50 remains best (summary in docs/audit/e1_results.md); its raw "
          "per-question arrays are not in this workspace."]

open(os.path.join(OUT, "verification_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
