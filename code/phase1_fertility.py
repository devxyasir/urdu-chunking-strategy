import json
import os
import random
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from tokenizers import Tokenizer

POOL = json.load(open("data/processed/pool_articles.json", encoding="utf-8"))
POOL_TITLES = [a["title"] for a in POOL["articles"]]

squad = json.load(open("data/raw/uqa/squad_train.json", encoding="utf-8"))
id2key, eng_order, eng = {}, {}, {}
for art in squad["data"]:
    t = art["title"]
    eng_order[t] = []
    for pi, p in enumerate(art["paragraphs"]):
        eng_order[t].append((t, pi))
        eng[(t, pi)] = p["context"]
        for qa in p["qas"]:
            id2key[qa["id"]] = (t, pi)

rows = pq.read_table("data/raw/uqa/uqa_train.parquet", columns=["id", "context"]).to_pylist()
groups = defaultdict(list)
for r in rows:
    k = id2key.get(r["id"])
    if k is not None:
        groups[k].append(r["context"])
chosen = {}
for k, ctxs in groups.items():
    cnt = Counter(ctxs)
    chosen[k] = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]

docs_ur = {t: "\n\n".join(chosen[k] for k in eng_order[t] if k in chosen) for t in POOL_TITLES}
docs_en = {t: "\n\n".join(eng[k] for k in eng_order[t]) for t in POOL_TITLES}

MODELS = {
    "bge-m3": ("data/raw/tokenizers/bge-m3.json", 8192, "sentence_bert_config max_seq_length"),
    "e5-large-instruct": ("data/raw/tokenizers/e5-large-instruct.json", 512, "tokenizer_config model_max_length"),
    "qwen3-0.6b": ("data/raw/tokenizers/qwen3-0.6b.json", 32768, "config max_position_embeddings"),
    "labse": ("data/raw/tokenizers/labse.json", 256, "sentence_bert_config max_seq_length"),
    "mpnet-multilingual": ("data/raw/tokenizers/mpnet-multilingual.json", 128, "sentence_bert_config max_seq_length=128 (excluded)"),
}
toks = {}
for k, v in MODELS.items():
    t = Tokenizer.from_file(v[0])
    t.no_truncation()
    t.no_padding()
    toks[k] = t

SIZES = [50, 100, 200, 400, 800]
random.seed(0)

lines = ["# Tokenizer fertility and truncation risk (measured, 120-article pool)", "",
         "Max sequence lengths were read from each model repository (`results/phase1/model_configs.txt`).",
         "Tokenizers were loaded with truncation explicitly disabled, so counts are not capped.", "",
         "## Whole-document fertility", "",
         "| model | vocab | max seq | Urdu tokens/word | English tokens/word | UR/EN token ratio |",
         "| --- | --- | --- | --- | --- | --- |"]

ur_words = sum(len(d.split()) for d in docs_ur.values())
en_words = sum(len(d.split()) for d in docs_en.values())
summary = {}
for name, (path, maxlen, src) in MODELS.items():
    tok = toks[name]
    ur_tok = sum(len(tok.encode(d).ids) for d in docs_ur.values())
    en_tok = sum(len(tok.encode(d).ids) for d in docs_en.values())
    summary[name] = {"ur_tok_per_word": ur_tok / ur_words, "en_tok_per_word": en_tok / en_words, "max_seq": maxlen}
    lines.append(f"| {name} | {tok.get_vocab_size()} | {maxlen} | {ur_tok/ur_words:.3f} | {en_tok/en_words:.3f} | {ur_tok/en_tok:.3f} |")

lines += ["", "## What a token-defined chunk means in words", "",
          "| model | 64 tok | 128 tok | 256 tok | 512 tok |", "| --- | --- | --- | --- | --- |"]
for name, (path, maxlen, src) in MODELS.items():
    f = summary[name]["ur_tok_per_word"]
    cells = [f"{t/f:.0f} words" for t in [64, 128, 256, 512]]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")

lines += ["", "## Window token counts by word size (Urdu pool, 3 word-aligned windows per article)", "",
          "| words | " + " | ".join(f"{n} median/p95/max" for n in MODELS) + " |",
          "| --- | " + " | ".join("---" for _ in MODELS) + " |"]
window_values = defaultdict(dict)
window_stats = defaultdict(dict)
for size in SIZES:
    per_model = defaultdict(list)
    for t in POOL_TITLES:
        words = docs_ur[t].split()
        if len(words) < size:
            continue
        for frac in (0.0, 0.4, 0.7):
            start = int(frac * (len(words) - size))
            chunk = " ".join(words[start:start + size])
            for name in MODELS:
                per_model[name].append(len(toks[name].encode(chunk).ids))
    cells = []
    for name in MODELS:
        v = sorted(per_model[name])
        med, p95 = v[len(v) // 2], v[int(len(v) * 0.95)]
        window_stats[size][name] = {"median": med, "p95": p95, "max": v[-1], "n": len(v)}
        window_values[size][name] = v
        cells.append(f"{med}/{p95}/{v[-1]}")
    lines.append(f"| {size} | " + " | ".join(cells) + " |")

lines += ["", "## Truncation share (windows exceeding the model max sequence length)", "",
          "| model | max seq | " + " | ".join(str(s) for s in SIZES) + " |",
          "| --- | --- | " + " | ".join("---" for _ in SIZES) + " |"]
for name, (path, maxlen, src) in MODELS.items():
    cells = []
    for size in SIZES:
        v = window_values[size][name]
        cells.append(f"{sum(1 for x in v if x > maxlen) / len(v):.1%}")
    lines.append(f"| {name} | {maxlen} | " + " | ".join(cells) + " |")

lines += ["", "Source of each max length: " + "; ".join(f"{k}={v[2]}" for k, v in MODELS.items())]
lines += ["", "mpnet-multilingual is reported for completeness but **excluded from the grid**: its",
          "sentence-transformers config truncates at 128 tokens, so every chunk above ~95 words would be cut."]

open("results/phase1/fertility.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({"summary": summary, "windows": {str(k): v for k, v in window_stats.items()}},
          open("results/phase1/fertility.json", "w", encoding="utf-8"), indent=2)
print("ok")
