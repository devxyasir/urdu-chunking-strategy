import hashlib
from collections import Counter

import pandas as pd

df = pd.read_parquet("data/raw/uqa/uqa_train.parquet")
lines = []

# size per article after dedup
df["wc"] = df["context"].str.split().str.len()
per = df.groupby("title").agg(rows=("id", "size"), ctx=("context", "nunique"), words=("wc", "sum"))
uniq = df.drop_duplicates(["title", "context"]).groupby("title")["wc"].sum()
per["uniq_words"] = uniq
lines.append("articles: " + str(len(per)))
lines.append(per.describe().to_string())
lines.append("")
lines.append("top 10 by unique paragraph count:")
lines.append(per.sort_values("ctx", ascending=False).head(10).to_string())

# inspect one article
title = df["title"].iloc[0]
sub = df[df["title"] == title]
lines.append("")
lines.append(f"article: {title.encode('unicode_escape').decode()} | rows={len(sub)} | unique contexts={sub['context'].nunique()}")
ctxs = sub["context"].drop_duplicates().tolist()
lines.append(f"unique context lengths: {[len(c) for c in ctxs[:10]]}")
for i, c in enumerate(ctxs[:3]):
    lines.append(f"ctx{i} sha={hashlib.sha1(c.encode()).hexdigest()[:8]} :: {c[:120].encode('unicode_escape').decode()}")

# are contexts near-duplicates?
import difflib
if len(ctxs) > 1:
    r = difflib.SequenceMatcher(None, ctxs[0], ctxs[1]).ratio()
    lines.append(f"similarity ctx0 vs ctx1: {r:.3f}")

# same paragraph across rows? group by question id
qid = sub["id"].iloc[0]
rows = sub[sub["id"] == qid]
lines.append(f"rows for one question id: {len(rows)}; unique contexts: {rows['context'].nunique()}")

# how many rows share exact context
vc = df["context"].value_counts()
lines.append("")
lines.append(f"contexts appearing once: {(vc == 1).sum():,}; max repeats: {vc.max():,}; mean repeats: {vc.mean():.2f}")
open("results/phase1/dup_check.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
