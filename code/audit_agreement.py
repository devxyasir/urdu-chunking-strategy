import csv
import json
import math
import os
from collections import Counter

P = "data/audit"
READERS = ["A", "B", "C"]
LABELS = ["0", "1", "?"]


def load(reader):
    rows = list(csv.DictReader(open(os.path.join(P, f"reader_{reader}.csv"), encoding="utf-8-sig")))
    return {r["qid"]: r["self_contained"].strip() for r in rows}


data = {r: load(r) for r in READERS}
qids = list(data["A"].keys())
assert all(len(data[r]) == 100 for r in READERS)
assert all(list(data[r].keys()) == qids for r in READERS)
assert all(v in LABELS for r in READERS for v in data[r].values())


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = Counter(a)
    pb = Counter(b)
    pe = sum((pa[k] / n) * (pb[k] / n) for k in LABELS)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss(rows):
    n = len(READERS)
    N = len(rows)
    counts = {k: 0 for k in LABELS}
    Pbar = 0.0
    totals = Counter()
    for q in rows:
        c = Counter(data[r][q] for r in READERS)
        for k in LABELS:
            counts[k] += c[k]
            totals[k] += c[k]
        Pbar += (sum(v * v for v in c.values()) - n) / (n * (n - 1))
    Pbar /= N
    pj = {k: totals[k] / (N * n) for k in LABELS}
    Pe = sum(v * v for v in pj.values())
    return Pbar, Pe, (Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


pairwise = {}
for i in range(3):
    for j in range(i + 1, 3):
        A, B = READERS[i], READERS[j]
        a = [data[A][q] for q in qids]
        b = [data[B][q] for q in qids]
        po, kappa = cohen(a, b)
        pairwise[f"{A}-{B}"] = {"agreement": po, "cohen_kappa": kappa}

row_counts = []
for q in qids:
    c = Counter(data[r][q] for r in READERS)
    top = c.most_common()
    if top[0][1] >= 2:
        label = top[0][0]
    else:
        label = "tie"
    row_counts.append((q, label, c))

maj = Counter(label for _, label, _ in row_counts)
maj1 = maj["1"]
lo, hi = wilson(maj1, len(qids))
per_reader = {r: Counter(data[r].values()) for r in READERS}
Pbar, Pe, fk = fleiss(qids)
unanimous = sum(1 for _, _, c in row_counts if max(c.values()) == 3)

out = {
    "n": len(qids),
    "per_reader": {r: {k: per_reader[r][k] for k in LABELS} for r in READERS},
    "pairwise": pairwise,
    "fleiss_kappa": fk,
    "fleiss_Pbar": Pbar,
    "fleiss_Pe": Pe,
    "unanimous": unanimous,
    "majority": {k: maj[k] for k in ["0", "1", "?", "tie"]},
    "majority_self_contained": {"k": maj1, "n": len(qids), "rate": maj1 / len(qids), "wilson_lo": lo, "wilson_hi": hi},
}
json.dump(out, open("docs/audit/self_contained_agreement.json", "w", encoding="utf-8"), indent=1)

lines = ["# Self-containedness audit: three readers", "",
         "100 questions, three raters, labels `1` self-contained, `0` context-dependent, `?` unclear.",
         "The raters worked independently on the same qid order.", "",
         "| Reader | self-contained (1) | context-dependent (0) | unclear (?) |", "| --- | --- | --- | --- |"]
for r in READERS:
    c = per_reader[r]
    lines.append(f"| {r} | {c['1']} | {c['0']} | {c['?']} |")
lines += ["", "## Agreement", "", "| Pair | Raw agreement | Cohen's kappa |", "| --- | --- | --- |"]
for k, v in pairwise.items():
    lines.append(f"| {k} | {v['agreement']:.1%} | {v['cohen_kappa']:.3f} |")
lines += ["", f"- Unanimous rows: **{unanimous}/100**.",
          f"- Fleiss' kappa (3 raters, 3 categories): **{fk:.3f}** (Pbar={Pbar:.3f}, Pe={Pe:.3f}).",
          "", "## Majority label", "",
          "| Label | Count |", "| --- | --- |"]
for k in ["1", "0", "?", "tie"]:
    lines.append(f"| {k} | {maj[k]} |")
lines += ["", f"Majority self-contained: **{maj1}/100 = {maj1/len(qids):.1%}**, "
              f"Wilson 95% CI **{lo:.1%}--{hi:.1%}**.",
          "", "A row is a tie only when all three labels differ; no tie was adjudicated by a fourth reader.",
          "", "Do not include the source CSVs in the public repository; they contain UQA text."]
open("docs/audit/self_contained_agreement.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(json.dumps(out, indent=1))
