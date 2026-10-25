import json
from collections import defaultdict

import pandas as pd

squad = json.load(open("data/raw/uqa/squad_train.json", encoding="utf-8"))
id2para = {}
para_words = {}
for art in squad["data"]:
    for pi, p in enumerate(art["paragraphs"]):
        key = (art["title"], pi)
        para_words[key] = len(p["context"].split())
        for qa in p["qas"]:
            id2para[qa["id"]] = key

uqa = pd.read_parquet("data/raw/uqa/uqa_train.parquet")
uqa["para_key"] = uqa["id"].map(id2para)
uqa["uwords"] = uqa["context"].str.split().str.len()
missing = uqa["para_key"].isna().sum()

g = uqa.groupby("para_key").agg(rows=("id", "size"), uctx=("context", "nunique"), uwords=("uwords", "mean"))
lines = [
    f"UQA train rows: {len(uqa):,}; rows with no English paragraph match: {missing:,}",
    f"English paragraphs covered: {len(g):,} of {len(para_words):,}",
    f"UQA rows per English paragraph: mean {g['rows'].mean():.2f}, median {g['rows'].median():.0f}, max {g['rows'].max()}",
    f"distinct Urdu contexts per English paragraph: mean {g['uctx'].mean():.2f}, median {g['uctx'].median():.0f}, max {g['uctx'].max()}",
    f"rows where distinct contexts == 1: {(g['uctx'] == 1).sum():,} ({(g['uctx'] == 1).mean():.2%})",
    "",
]

# compare lengths for a sample
sample = g.sample(min(500, len(g)), random_state=0)
eng = [para_words[k] for k in sample.index]
lines.append(f"sample of {len(sample)} English paragraphs: English words mean {sum(eng)/len(eng):.1f}; Urdu words mean {sample['uwords'].mean():.1f}")

# one paragraph with many rows
k = g["rows"].idxmax()
sub = uqa[uqa["para_key"] == k]
lines.append("")
lines.append(f"busiest English paragraph {k}: rows={len(sub)}, distinct contexts={sub['context'].nunique()}, English words={para_words[k]}, Urdu words={sub['uwords'].iloc[0]}")
ctxs = sub["context"].drop_duplicates().tolist()
lines.append("first 3 context prefixes:")
for c in ctxs[:3]:
    lines.append("  " + c[:100].encode("unicode_escape").decode())
import difflib
if len(ctxs) > 1:
    lines.append(f"similarity: {difflib.SequenceMatcher(None, ctxs[0], ctxs[1]).ratio():.3f}")

open("results/phase1/para_map.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
