import json
import os
import re
import unicodedata
from collections import Counter

import pandas as pd

OUT = "results/phase1"
os.makedirs(OUT, exist_ok=True)
URDU_PUNCT = {"\u06d4": "۔", "\u061f": "؟"}
DIACRITICS = set(range(0x064B, 0x0653)) | {0x0670, 0x0653, 0x0654, 0x0655}
VARIANTS = {
    "\u064a": "ARABIC YEH ي", "\u06cc": "FARSI YEH ی",
    "\u0643": "ARABIC KAF ك", "\u06a9": "KEHEH ک",
    "\u06c1": "HEH GOAL ہ", "\u06be": "HEH DOACHASHMEE ھ",
    "\u0629": "TEH MARBUTA ة", "\u06c3": "TEH MARBUTA GOAL ۃ",
    "\u0640": "TATWEEL ـ",
}

lines = ["# Corpus statistics — UQA (measured)", ""]


def block(name, df):
    n_rows = len(df)
    qids = df["id"].nunique()
    answerable_rows = int((~df["is_impossible"]).sum())
    answerable_q = df.loc[~df["is_impossible"], "id"].nunique()
    titles = df["title"].nunique()
    paras = df.drop_duplicates(["title", "context"]).shape[0]
    words = int(df["context"].str.split().str.len().sum())
    uniq_ctx_words = int(df.drop_duplicates("context")["context"].str.split().str.len().sum())
    ctx_len = df.drop_duplicates("context")["context"].str.split().str.len()
    lines.extend([
        f"## {name}",
        f"- rows (one per answer variant): {n_rows:,}",
        f"- unique question ids: {qids:,}",
        f"- answerable rows / unique answerable questions: {answerable_rows:,} / {answerable_q:,}",
        f"- unique articles (title): {titles:,}",
        f"- unique paragraphs: {paras:,}",
        f"- total context words (with repeats): {words:,}",
        f"- total words in unique paragraphs: {uniq_ctx_words:,}",
        f"- words per unique paragraph: median {ctx_len.median():.2f}, mean {ctx_len.mean():.2f}, p90 {ctx_len.quantile(0.9):.0f}, max {ctx_len.max()}",
        "",
    ])


tr = pd.read_parquet("data/raw/uqa/uqa_train.parquet")
va = pd.read_parquet("data/raw/uqa/uqa_validation.parquet")
block("train", tr)
block("validation", va)

lines.append("## Answer-span audit (all rows, not a sample)")
for name, df in [("train", tr), ("validation", va)]:
    a = df[~df["is_impossible"]].copy()
    starts = a["answer_start"].astype(int).to_numpy()
    texts = a["answer"].astype(str).to_numpy()
    ctxs = a["context"].astype(str).to_numpy()
    ok = sum(1 for c, s, t in zip(ctxs, starts, texts) if c[s:s + len(t)] == t)
    fail = len(a) - ok

    def norm(s):
        return re.sub(r"\s+", " ", s).strip()

    ok_norm = sum(1 for c, s, t in zip(ctxs, starts, texts) if norm(c)[norm(c).find(norm(t)):][:len(norm(t))] == norm(t) and norm(t) in norm(c))
    short = sum(1 for t in texts if len(t.split()) <= 1)
    lines.extend([
        f"- {name}: answerable rows {len(a):,}; exact span match {ok:,} ({ok / len(a):.4%}); mismatches {fail:,}",
        f"- {name}: answer text found after whitespace normalisation: {ok_norm:,} ({ok_norm / len(a):.4%})",
        f"- {name}: single-token answers: {short:,} ({short / len(a):.2%})",
    ])

lines.append("")
lines.append("## Unicode statistics (unique contexts, validation split)")

ctxs = va.drop_duplicates("context")["context"].astype(str).tolist()
text = "\n".join(ctxs)
chars = Counter(text)
total = len(text)
classes = Counter()
for ch in text:
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        classes["Arabic-script block"] += 1
    elif ch.isalpha():
        classes["other letter"] += 1
    elif ch.isdigit():
        classes["digit"] += 1
    elif ch.isspace():
        classes["space"] += 1
    else:
        classes["other (punct/symbol/mark)"] += 1
lines.append(f"- characters: {total:,} across {len(ctxs):,} unique paragraphs")
for k, v in classes.most_common():
    lines.append(f"- {k}: {v:,} ({v / total:.4%})")
lines.append("")
lines.append("### Urdu-specific characters present")
for ch in ["\u06d4", "\u061f", "!", "\u060c", "\u061b", ",", "\u066b", "\u066c", "\u200c", "\u200d",
           "\u0640", "\u0670", "\u064e", "\u0650", "\u064f", "\u0652", "\u0651", "\u064e"]:
    c = chars.get(ch, 0)
    lines.append(f"- U+{ord(ch):04X} {unicodedata.name(ch, '?'):40s} count={c:,} ({c / total:.5%})")
lines.append("")
lines.append("### Letter variants (confusable forms)")
for ch, label in VARIANTS.items():
    c = chars.get(ch, 0)
    lines.append(f"- {label}: {c:,} ({c / total:.5%})")
lines.append("")
lines.append("### Digits and separators")
for label, chs in [("ASCII 0-9", "0123456789"), ("Urdu ۰-۹", "۰۱۲۳۴۵۶۷۸۹"), ("Arabic-Indic ٠-٩", "٠١٢٣٤٥٦٧٨٩")]:
    c = sum(chars.get(ch, 0) for ch in chs)
    lines.append(f"- {label}: {c:,}")
for label, ch in [("ASCII space", " "), ("NBSP", "\u00a0"), ("ZWNJ", "\u200c"), ("ZWJ", "\u200d"), ("RLM", "\u200f"), ("BOM", "\ufeff")]:
    lines.append(f"- {label}: {chars.get(ch, 0):,}")
lines.append("")
lines.append("### Sentence-boundary candidates")
for ch in ["\u06d4", "\u061f", "!", ".", "\u061b"]:
    lines.append(f"- U+{ord(ch):04X}: {chars.get(ch, 0):,}")
lines.append("")
lines.append("### Composing vs precomposed forms")
lines.append(f"- NFC-equal to raw: {unicodedata.normalize('NFC', text) == text}")
lines.append(f"- combining marks (Mn): {sum(1 for ch in text if unicodedata.category(ch) == 'Mn'):,}")

open(os.path.join(OUT, "corpus_stats.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

summary = {
    "train_rows": int(len(tr)), "train_qids": int(tr["id"].nunique()),
    "train_articles": int(tr["title"].nunique()),
    "val_rows": int(len(va)), "val_qids": int(va["id"].nunique()),
    "val_articles": int(va["title"].nunique()),
    "val_unique_paragraphs": len(ctxs),
    "val_unique_paragraph_words": int(sum(len(c.split()) for c in ctxs)),
    "chars": total,
}
json.dump(summary, open(os.path.join(OUT, "corpus_stats.json"), "w", encoding="utf-8"), indent=2)
print("ok")
