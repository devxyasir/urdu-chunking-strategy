import csv
import hashlib
import json
import os

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


files = ["prereg.md", "design.md", "docs/cost_table.md", "docs/phase_1_notes.md",
         "docs/annotation/README.md", "docs/annotation/datasheet.md",
         "docs/annotation/article_pool.csv", "docs/annotation/native_worksheet.csv",
         "docs/annotation/native_reader.html", "docs/annotation/native_template.jsonl",
         "code/validate_native.py", "results/phase1/power.md", "results/phase1/cost.md",
         "results/phase1/fertility.md", "results/phase1/chunk_counts.md",
         "results/phase1/pool_stats.md", "results/phase1/corpus_stats.md",
         "results/phase1/prereg_hash.txt"]
for f in files:
    check(f, os.path.exists(f), f"{os.path.getsize(f):,} bytes" if os.path.exists(f) else "missing")

with open("docs/annotation/article_pool.csv", encoding="utf-8-sig") as f:
    pool_rows = list(csv.DictReader(f))
check("article pool = 60", len(pool_rows) == 60, str(len(pool_rows)))
check("all pool rows have revid", all(r["revid"] for r in pool_rows), "")

with open("docs/annotation/native_worksheet.csv", encoding="utf-8-sig") as f:
    ws = list(csv.DictReader(f))
check("worksheet = 300 rows", len(ws) == 300, str(len(ws)))
check("worksheet starts blank", all(not (r["question"] or "").strip() for r in ws), "")

pool = json.load(open("data/processed/pool_articles.json", encoding="utf-8"))
check("parallel pool = 150 articles", pool["totals"]["articles"] == 150, str(pool["totals"]))
check("questions = 9077", pool["totals"]["questions"] == 9077, str(pool["totals"]["questions"]))
test_articles = sum(1 for a in pool["articles"] if a["split"] == "test")
check("test articles = 120", test_articles == 120, str(test_articles))

h = hashlib.sha256(open("prereg.md", "rb").read()).hexdigest()
recorded = open("results/phase1/prereg_hash.txt", encoding="utf-8").read()
check("prereg hash matches", h in recorded, h[:16])

check("no Phase 5 outputs", not os.path.exists("results/phase5"), "")
check("no test-split metrics", not os.path.exists("results/metrics"), "")

lines = ["# Phase 1 completion check", ""]
for name, ok, detail in checks:
    lines.append(f"- {'PASS' if ok else 'FAIL'} | {name} | {detail}")
lines.append("")
lines.append(f"{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
open("results/phase1/final_check.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[-3:]))
