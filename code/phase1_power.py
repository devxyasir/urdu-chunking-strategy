import json

import numpy as np

POOL = json.load(open("data/processed/pool_articles.json", encoding="utf-8"))
test_q = np.array([a["questions"] for a in POOL["articles"] if a["split"] == "test"], dtype=float)
dev_q = np.array([a["questions"] for a in POOL["articles"] if a["split"] == "dev"], dtype=float)
A = len(test_q)
rng = np.random.default_rng(20261024)

lines = [
    "# Power analysis (simulation, pre-data)",
    "",
    f"- test articles: {A}; test questions: {int(test_q.sum()):,}",
    f"- questions per test article: median {np.median(test_q):.0f}, mean {test_q.mean():.1f}, min {test_q.min():.0f}, max {test_q.max():.0f}",
    f"- dev articles: {len(dev_q)}; dev questions: {int(dev_q.sum()):,}",
    "",
    "Model: per-question paired difference d_ai = delta_a + eps_ai, delta_a ~ N(delta, tau^2),",
    "eps_ai ~ N(0, sigma^2). Primary test: one-sample t-test on the 120 article means (cluster unit),",
    "alpha = 0.05 two-sided. 2,000 simulations per cell, seed 20261024.",
    "",
    "## Power at delta = 0.05 (5 recall points), 120 test articles",
    "",
    "| sigma (question noise) | tau=0.00 | tau=0.10 | tau=0.20 | tau=0.30 |",
    "| --- | --- | --- | --- | --- |",
]


def simulate(delta, sigma, tau, reps=2000, alpha=0.05):
    hits = 0
    for _ in range(reps):
        d_a = rng.normal(delta, tau, size=A)
        m = test_q
        mean = d_a + rng.normal(0, sigma / np.sqrt(m))
        se = mean.std(ddof=1) / np.sqrt(A)
        t = mean.mean() / se
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df=A - 1)
        hits += p < alpha
    return hits / reps


def mde(sigma, tau, target=0.8):
    lo, hi = 0.0, 0.4
    for _ in range(18):
        mid = (lo + hi) / 2
        if simulate(mid, sigma, tau, reps=400) < target:
            lo = mid
        else:
            hi = mid
    return hi


for sigma in [0.20, 0.30, 0.40]:
    cells = [f"{simulate(0.05, sigma, tau):.2f}" for tau in [0.0, 0.10, 0.20, 0.30]]
    lines.append(f"| {sigma:.2f} | " + " | ".join(cells) + " |")

lines += ["", "## Minimum detectable effect (80% power, 120 articles)", "",
          "| sigma | tau=0.00 | tau=0.10 | tau=0.20 | tau=0.30 |",
          "| --- | --- | --- | --- | --- |"]
for sigma in [0.20, 0.30, 0.40]:
    cells = [f"{mde(sigma, tau):.3f}" for tau in [0.0, 0.10, 0.20, 0.30]]
    lines.append(f"| {sigma:.2f} | " + " | ".join(cells) + " |")

# naive question-level comparison
lines += ["", "## Inflation if questions are treated as independent (delta=0.05, sigma=0.30)", ""]
for label, n in [("120 articles (cluster means)", A), ("all test questions", int(test_q.sum()))]:
    hits = 0
    for _ in range(2000):
        d = rng.normal(0.05, 0.30, size=n)
        se = d.std(ddof=1) / np.sqrt(n)
        from scipy import stats
        hits += 2 * stats.t.sf(abs(d.mean() / se), df=n - 1) < 0.05
    lines.append(f"- {label}: power {hits/2000:.3f}")

lines += ["", "The question-level figure is an artefact: the 61 questions per article share one chunk set,",
          "so they are not independent. The article-cluster analysis is the one that governs the claims."]

lines += ["", "## Config-ranking correlation (H6, exploratory)", "",
          "With 36 configurations, the 95% CI for a Spearman rho of 0.7 spans roughly 0.45-0.85 "
          "(Fisher z, n=36). H6 is therefore a directional check, not a precise estimate."]

open("results/phase1/power.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
