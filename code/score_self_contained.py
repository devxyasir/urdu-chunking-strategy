import csv
import math
import sys

P = "data/external/urdu_chunking_full/audit_100_questions.csv"
rows = list(csv.DictReader(open(P, encoding="utf-8-sig")))
if not rows:
    sys.exit("empty audit file")
blank = [r for r in rows if not r.get("self_contained", "").strip()]
bad = [r for r in rows if r.get("self_contained", "").strip() not in ("0", "1", "?")]
print(f"rows={len(rows)} blank={len(blank)} bad_values={len(bad)}")
if blank or bad:
    print("Fill every row with 0, 1, or ? before scoring.")
    sys.exit(1)

vals = [r["self_contained"].strip() for r in rows]
n = len(vals)
k = sum(v == "1" for v in vals)
p = k / n
z = 1.959963984540054
den = 1 + z * z / n
centre = (p + z * z / (2 * n)) / den
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
print(f"self-contained: {k}/{n} = {p:.3f} (Wilson 95% CI {max(0, centre-half):.3f}--{min(1, centre+half):.3f})")
print(f"context-dependent: {sum(v == '0' for v in vals)}; unclear: {sum(v == '?' for v in vals)}")
