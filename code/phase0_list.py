import csv
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else ""
rows = list(csv.DictReader(open("results/phase0/candidates.csv", encoding="utf-8")))
for r in rows:
    if not tag or r["tag"] == tag:
        print(r["arxiv_id"], "|", r["published"], "|", r["title"][:115])
