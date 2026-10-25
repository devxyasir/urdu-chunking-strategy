import json

CC = json.load(open("results/phase1/chunk_counts.json", encoding="utf-8"))
FERT = json.load(open("results/phase1/fertility.json", encoding="utf-8"))
WS = FERT["windows"]
SUMMARY = FERT["summary"]

fixed = {int(k): {float(o): v for o, v in d.items()} for k, d in CC["fixed"].items()}
sent = {int(k): {float(o): v for o, v in d.items()} for k, d in CC["sentence"].items()}
TOTAL_WORDS = CC["total_words"]


def token_count(model, budget):
    return max(1, round(TOTAL_WORDS * SUMMARY[model]["ur_tok_per_word"] / budget))
SIZES = [50, 100, 200]
OVS = [0.0, 0.10, 0.25]
MODELS = ["bge-m3", "e5-large-instruct", "qwen3-0.6b"]
DIM = {"bge-m3": 1024, "e5-large-instruct": 1024, "qwen3-0.6b": 1024}
THROUGHPUTS = [2500, 5000, 10000]


def median_tokens(model, size):
    return WS[str(size)][model]["median"]


def fixed_token_words(model, budget):
    return max(1, round(budget / SUMMARY[model]["ur_tok_per_word"]))


lines = ["# Cost and compute table (measured chunk counts, declared throughput assumptions)", "",
         "Pool: 150 articles, 1,007,122 Urdu words, 6,720 paragraphs, 9,077 answerable questions.",
         "Dense grid per model: 21 configs (fixed-word 3 sizes x 3 overlaps = 9; token-fixed 3;",
         "sentence-aware 3; recursive-English 3; recursive-Urdu 3, the last four at overlap 0).",
         "BM25 gets the full 75-config grid (5 strategies x 5 sizes x 3 overlaps) and 3 normalization",
         "variants; it is CPU-only and its cost is reported separately as index builds.",
         "", "## Passages and tokens per arm", "",
         "| arm | model | configs | passages | content tokens |",
         "| --- | --- | --- | --- | --- |"]

arms = []


def urdu_primary(model, english=False):
    n = 0
    tokens = 0.0
    cfg = 0
    for size in SIZES:
        for ov in OVS:
            c = fixed[size][ov]
            n += c
            tokens += c * median_tokens(model, size)
            cfg += 1
    for m2, budget in [("bge-m3", 64), ("bge-m3", 128), ("bge-m3", 256)]:
        pass
    budgets = [64, 128, 256]
    for b in budgets:
        c = token_count(model, b)
        n += c
        tokens += c * b
        cfg += 1
    for size in SIZES:
        c = sent[size][0.0]
        n += c
        tokens += c * median_tokens(model, size) * 0.9
        cfg += 1
    for size in SIZES:
        c = int(fixed[size][0.0] * 1.10)
        n += c
        tokens += c * median_tokens(model, size)
        cfg += 1
        c = int(fixed[size][0.0] * 1.10)
        n += c
        tokens += c * median_tokens(model, size)
        cfg += 1
    return cfg, n, tokens


for m in MODELS:
    cfg, n, t = urdu_primary(m)
    arms.append(("Urdu primary", m, cfg, n, t))

for m in ["bge-m3", "qwen3-0.6b"]:
    cfg = 0
    n = 0
    t = 0.0
    for size in [400, 800]:
        for ov in [0.0, 0.25]:
            c = fixed[size][ov]
            n += c
            t += c * median_tokens(m, size)
            cfg += 1
        c = sent[size][0.0]
        n += c
        t += c * median_tokens(m, size)
        cfg += 1
        c = int(fixed[size][0.0] * 1.10)
        n += c
        t += c * median_tokens(m, size)
        cfg += 1
    arms.append(("Urdu extended 400/800", m, cfg, n, t))

# English arm: bge-m3 only, 21 configs, English fertility 1.486
for m in ["bge-m3"]:
    cfg = 0
    n = 0
    t = 0.0
    for size in SIZES:
        for ov in OVS:
            c = fixed[size][ov]
            n += c
            t += c * size * SUMMARY[m]["en_tok_per_word"] * 1.05
            cfg += 1
    for b in [64, 128, 256]:
        c = token_count(m, b)
        n += c
        t += c * b
        cfg += 1
    for size in SIZES:
        for fn, factor in [(sent, 0.9), (fixed, 1.10), (fixed, 1.10)]:
            c = int(fn[size][0.0] * factor)
            n += c
            t += c * size * SUMMARY[m]["en_tok_per_word"]
            cfg += 1
    arms.append(("English control", m, cfg, n, t))

# Native arm: provisional 60 articles x 3000 words = 180k words (0.179 of the Urdu pool)
scale = 180000 / CC["total_words"]
for m in ["bge-m3", "qwen3-0.6b"]:
    cfg, n, t = urdu_primary(m)
    arms.append(("Native (provisional)", m, cfg, int(n * scale), t * scale))

for arm, m, cfg, n, t in arms:
    lines.append(f"| {arm} | {m} | {cfg} | {n:,} | {t/1e6:.2f}M |")

total_tokens = sum(a[4] for a in arms)
total_passages = sum(a[3] for a in arms)
lines += ["", f"**Total passages to embed: {total_passages:,}. Total content tokens: {total_tokens/1e6:.1f}M.**", ""]

lines += ["## GPU hours by measured-throughput assumption", "",
          "| tokens/s | GPU hours | fits 30 h? |", "| --- | --- | --- |"]
for thr in THROUGHPUTS:
    h = total_tokens / thr / 3600
    lines.append(f"| {thr:,} | {h:.1f} | {'yes' if h <= 30 else 'no'} |")

lines += ["", "The throughput figure is *not* assumed silently: Phase 4 measures it on the actual GPU",
          "with 200 passages and replaces this table. If the measured rate is below 2,500 tokens/s,",
          "the declared cut order applies (drop API embedders - already dropped; then overlap factor;",
          "then Urdu MS MARCO; then the 4B model - never added)."]

lines += ["", "## BM25 and storage", ""]
bm25_builds = 75 * 3
lines.append(f"- BM25 index builds (75 configs x 3 normalization variants): {bm25_builds}. Each build")
lines.append("  over ~1M words takes seconds to a few minutes on CPU; budget 2-3 CPU-hours for the full set.")
lines.append(f"- Raw BM25 runs: 225 configs x 9,077 queries; CPU-only, budget under 1 hour.")
lines.append(f"- Embedding storage: {total_passages:,} passages x {2} bytes x 1024 dims = "
             f"{total_passages*1024*2/1e9:.2f} GB of float16 vectors (plus query vectors), before deduplication by cache key.")
lines.append("- API spend: **USD 0**. The optional API embedders are dropped at GATE 1 (first item in the cut order).")

json.dump({"arms": [{"arm": a, "model": m, "configs": c, "passages": n, "tokens": t} for a, m, c, n, t in arms],
           "total_passages": total_passages, "total_tokens": total_tokens},
          open("results/phase1/cost.json", "w", encoding="utf-8"), indent=2)
open("results/phase1/cost.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok", total_passages, round(total_tokens / 1e6, 1))
