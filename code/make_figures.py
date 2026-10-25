import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = Path(r"C:\Users\HP\AppData\Roaming\com.ai4s.workbench\runtime\xdg-config\opencode\skills\publication-figures\openscience.mplstyle")
plt.style.use(str(STYLE))
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
A = Path("results/analysis")

COL = {
    "bge-m3": "#2a78d6",
    "e5-large-instruct": "#1baf7a",
    "Qwen3-0.6B": "#eda100",
    "LaBSE": "#008300",
    "BM25-ws": "#4a3aa7",
    "BM25-char": "#e34948",
}
LABEL = {"bge-m3": "bge-m3", "e5-large-instruct": "e5-large", "Qwen3-0.6B": "Qwen3-0.6B",
         "LaBSE": "LaBSE", "BM25-ws": "BM25", "BM25-char": "BM25-char"}


def read(name):
    with open(A / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x)


words = read("table_words.csv")
tokens = read("table_tokens.csv")
sdiffs = read("table_strategy_diffs.csv")
overlap = read("table_overlap.csv")
english = read("table_english.csv")
retr = read("table_retrievers.csv")
fert = json.load(open("data/external/urdu_chunking_full/tokenizer_fertility.json", encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------- Fig 1: size curves in words ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for m in ["bge-m3", "e5-large-instruct", "Qwen3-0.6B", "LaBSE", "BM25-ws"]:
    rows = sorted([r for r in words if r["model"] == m], key=lambda r: int(r["size"]))
    if not rows:
        continue
    x = [int(r["size"]) for r in rows]
    y = [fnum(r["mean"]) for r in rows]
    lo = [fnum(r["lo"]) for r in rows]
    hi = [fnum(r["hi"]) for r in rows]
    ls = "--" if m == "e5-large-instruct" else "-"
    ax.plot(x, y, marker="o", color=COL[m], label=LABEL[m], linestyle=ls, zorder=3)
    ax.fill_between(x, lo, hi, color=COL[m], alpha=0.12, linewidth=0, zorder=2)
ax.set_xscale("log", base=2)
ax.set_xticks([50, 100, 200, 400, 800])
ax.set_xticklabels(["50", "100", "200", "400", "800"])
ax.set_xlabel("Chunk size (words)")
ax.set_ylabel("BM-Recall@500 (crossing)")
ax.set_title("Word-defined chunk size")
ax.legend(ncol=2, loc="lower left")
ax.set_ylim(0.22, 0.85)
save(fig, "fig1_word_size_curves")

# ---------------- Fig 2: tokenizer fertility and token budgets ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax = axes[0]
models = ["bge-m3", "e5-large-instruct", "Qwen3-0.6B", "LaBSE"]
ur = [fert["bge"]["ur"], fert["e5"]["ur"], fert["qwen"]["ur"], fert["labse"]["ur"]]
en = [fert["bge"]["en"], fert["e5"]["en"], fert["qwen"]["en"], fert["labse"]["en"]]
x = np.arange(len(models))
w = 0.36
ax.bar(x - w / 2, ur, w, color="#2a78d6", label="Urdu")
ax.bar(x + w / 2, en, w, color="#9ec5f4", label="English")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[m] for m in models], rotation=0)
ax.set_ylabel("Tokens per word")
ax.set_title("(a) Tokenizer fertility")
ax.legend(loc="upper left")
ax.set_ylim(0, 3.6)
for i, v in enumerate(ur):
    ax.text(i - w / 2, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#2a2723")

ax = axes[1]
for m in ["bge-m3", "e5-large-instruct", "Qwen3-0.6B"]:
    rows = sorted([r for r in tokens if r["model"] == m], key=lambda r: int(r["token_budget"]))
    xs = [int(r["token_budget"]) for r in rows]
    ys = [fnum(r["mean"]) for r in rows]
    lo = [fnum(r["lo"]) for r in rows]
    hi = [fnum(r["hi"]) for r in rows]
    ax.plot(xs, ys, marker="o", color=COL[m], label=LABEL[m])
    ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.12, linewidth=0)
ax.set_xscale("log", base=2)
ax.set_xticks([64, 128, 256])
ax.set_xticklabels(["64", "128", "256"])
ax.set_xlabel("Chunk size (model tokens)")
ax.set_ylabel("BM-Recall@500 (crossing)")
ax.set_title("(b) Token-defined chunk size")
ax.legend(loc="lower right")
ax.set_ylim(0.42, 0.82)
ax.text(0.02, 0.97, "128 tokens = 96 words (XLM-R) / 42 words (Qwen3)",
        transform=ax.transAxes, fontsize=7.6, color="#5a554d", va="top")
fig.tight_layout()
save(fig, "fig2_token_units")

# ---------------- Fig 3: Urdu-aware vs English default ----------------
fig, ax = plt.subplots(figsize=(3.7, 3.6))
sizes = [50, 100, 200]
for j, m in enumerate(["BM25-ws", "bge-m3", "e5-large-instruct", "Qwen3-0.6B"]):
    xs, ys, lo, hi = [], [], [], []
    for i, s in enumerate(sizes):
        row = [r for r in sdiffs if r["model"] == m and int(r["size"]) == s
               and r["contrast"] == "sentence_aware - recursive_english"]
        if not row:
            continue
        xs.append(i + (j - 1.5) * 0.08)
        ys.append(fnum(row[0]["diff"]) * 100)
        lo.append(fnum(row[0]["lo"]) * 100)
        hi.append(fnum(row[0]["hi"]) * 100)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                fmt="o", color=COL[m], label=LABEL[m], capsize=2, markersize=5)
ax.axhline(0, color="#8c877d", linewidth=0.8, zorder=1)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(["50", "100", "200"])
ax.set_xlabel("Chunk size (words)")
ax.set_ylabel("Δ BM-Recall@500 (pts)", fontsize=9)
ax.set_title("Urdu-aware − English default", fontsize=10)
ax.tick_params(labelsize=8.5)
ax.legend(ncol=1, loc="lower left", fontsize=7.5)
save(fig, "fig3_urdu_vs_english_splitting")

# ---------------- Fig 4: overlap effect ----------------
fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
panels = [("bge-m3", axes[0][0]), ("e5-large-instruct", axes[0][1]),
          ("Qwen3-0.6B", axes[1][0]), ("BM25-ws", axes[1][1])]
colors = {50: "#2a78d6", 100: "#1baf7a", 200: "#eda100"}
for m, ax in panels:
    for s in [50, 100, 200]:
        rows = sorted([r for r in overlap if r["model"] == m and int(r["size"]) == s],
                      key=lambda r: fnum(r["overlap"]))
        x = [fnum(r["overlap"]) * 100 for r in rows]
        y = [fnum(r["mean"]) for r in rows]
        lo = [fnum(r["lo"]) for r in rows]
        hi = [fnum(r["hi"]) for r in rows]
        ax.plot(x, y, marker="o", color=colors[s], label=f"{s} words")
        ax.fill_between(x, lo, hi, color=colors[s], alpha=0.12, linewidth=0)
    ax.set_title(LABEL[m])
    ax.set_xticks([0, 10, 25])
    ax.set_xticklabels(["0%", "10%", "25%"])
    ax.set_ylim(0.45, 0.92)
for ax in axes[1]:
    ax.set_xlabel("Overlap")
for ax in axes[:, 0]:
    ax.set_ylabel("BM-Recall@500 (crossing)")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Overlap helps at every size and model", y=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig4_overlap")

# ---------------- Fig 5: English vs Urdu ----------------
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
for ax, strat, title in [(axes[0], "fixed_words", "(a) Fixed words"),
                         (axes[1], "recursive_english", "(b) Recursive English")]:
    for lang, color, marker in [("Urdu", "#2a78d6", "o"), ("English", "#e34948", "s")]:
        rows = sorted([r for r in english if r["language"] == lang and r["strategy"] == strat],
                      key=lambda r: int(r["size"]))
        x = [int(r["size"]) for r in rows]
        y = [fnum(r["mean"]) for r in rows]
        lo = [fnum(r["lo"]) for r in rows]
        hi = [fnum(r["hi"]) for r in rows]
        ax.plot(x, y, marker=marker, color=color, label=lang)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([50, 100, 200])
    ax.set_xticklabels(["50", "100", "200"])
    ax.set_xlabel("Chunk size (words)")
    ax.set_title(title)
    ax.legend(loc="lower left")
axes[0].set_ylabel("BM-Recall@500 (crossing)")
axes[0].set_ylim(0.55, 0.97)
fig.suptitle("English is easier; fixed-word optimum is the same (bge-m3)", y=1.02)
fig.tight_layout()
save(fig, "fig5_english_vs_urdu")

# ---------------- Fig 6: best systems ----------------
best = {}
for r in retr:
    if int(r["size"]) == 50 and r["strategy"] == "sentence_aware":
        best[r["system"]] = (fnum(r["mean"]), fnum(r["lo"]), fnum(r["hi"]))
order = sorted(best, key=lambda k: best[k][0])
fig, ax = plt.subplots(figsize=(6.8, 3.6))
y = np.arange(len(order))
for i, k in enumerate(order):
    m, lo, hi = best[k]
    color = COL["BM25-ws"] if k.startswith("BM25") else ("#5a554d" if "RRF" in k else COL.get(k.split(" + ")[0], "#2a78d6"))
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.4, solid_capstyle="round", alpha=0.45)
    ax.plot([m], [i], "o", color=color, markersize=7)
    ax.text(hi + 0.006, i, f"{m:.3f}", va="center", fontsize=8, color="#2a2723")
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_xlabel("BM-Recall@500 (crossing), sentence-aware 50 words")
ax.set_xlim(0.66, 0.92)
ax.set_title("Best configuration per system")
save(fig, "fig6_best_systems")

print("done")
