# Reproducibility plan

The current workspace is a private research repository. It contains UQA-derived text
(`data/processed/corpus_ur.jsonl`, `data/external/urdu_chunking_full/corpus_ur.jsonl`,
`questions.csv`) and is not suitable for a public artifact as-is. This plan separates what can be
public from what cannot.

## What exists now

| Item | Path | Public? |
| --- | --- | --- |
| Code (corpus build, chunkers, BM25, analysis, figures) | `code/` | Yes, after a licence is chosen |
| Frozen grids and paper lists | `configs/` | Yes |
| Run index (one row per run, config only) | `data/external/urdu_chunking_full/run_index.jsonl` | Yes |
| Per-question metrics (no question text) | `data/external/urdu_chunking_full/runs.jsonl.gz` (12.5 MB) | Yes, with question IDs |
| Top-10 ranked IDs | `data/external/urdu_chunking_full/rankings_*.npz` | Yes, no text |
| Manifest, aux, summary | `data/external/urdu_chunking_full/manifest.json`, `aux_.json`, `summary.json` | Yes |
| Analysis tables and report | `results/analysis/` | Yes |
| Figures | `figures/` | Yes |
| Paper source and PDFs | `paper/`, `guide/` | Yes (paper licence) |
| UQA-derived corpora and question text | `data/processed/`, `data/external/urdu_chunking_full/corpus_*.jsonl`, `questions.csv` | **No** without UQA permission |
| Third-party PDFs and metadata | `refs/` | Metadata yes; PDFs only if licence permits |

`runs.jsonl.gz` contains per-question arrays keyed by metric, not question text. It is 12.5 MB and
fits a normal GitHub repository. The uncompressed `runs.jsonl` is 110 MB and is git-ignored; do not
commit it.

## Required changes before a public release

1. **Create a public artifact directory** (or a new public repository) with the public items above
   and **no** `corpus_ur.jsonl`, `corpus_en.jsonl`, or `questions.csv`.
2. Add `question_ids.csv` with columns `qid,article_id,split` and no question/answer text. The
   per-question arrays are aligned to this order. Verify the order against the analysis script.
3. Add `scripts/fetch_data.py` that downloads SQuAD2.0 and UQA from their original URLs and rebuilds
   the corpus. The existing `code/phase1_download_data.py` is a starting point; it must be tested
   from a clean directory.
4. Add `requirements-full.txt` with the GPU stack: `torch`, `sentence-transformers`,
   `transformers`, `huggingface_hub`, plus `requirements.txt`. Pin versions used:
   `sentence-transformers 6.2.0`, PyTorch (record the exact CUDA build), `transformers` (record),
   `numpy`, `scipy`.
5. Add `LICENSE` for the code (MIT or Apache-2.0; the author must choose) and a separate paper
   licence (CC BY 4.0 is conventional). Do not apply a code licence to UQA text.
6. Add `DATA.md`: UQA has no explicit licence found; users must obtain it from the UQA authors and
   cite Arif et al. (2024). SQuAD2.0 is CC BY-SA 4.0. State the date of the licence check.
7. Add `MANIFEST.md` mapping each paper table/figure to a script and a run ID. The existing
   `run_index.jsonl` is the machine-readable source; the missing piece is a human-readable table.
8. Add a smoke test that runs on 2 articles without a GPU: corpus build, chunking, BM25, metrics,
   and one table cell. The notebook has self-tests; expose them as `pytest` tests.
9. Add a `CITATION.cff` with the paper title, author, year, and repository URL. Keep the URL private
   until the artifact is public.
10. Remove or gate the private GitHub URL in `README.md` and `CITATION.cff` until the repository is
    public.

## Commands that must work from a clean checkout

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/phase1_download_data.py
python code/build_corpus.py
python code/analyze_full.py
python code/analyze_extra.py
python code/make_figures.py
```

The GPU reproduction is a separate path:

```bash
pip install -r requirements-full.txt
jupyter notebook notebooks/urdu_chunking_fullrun.ipynb
# set USE_DRIVE=False, STAGES=["A","B","C","D","E"], RUN_SPLITS=["test"]
```

The notebook is resumable and writes a result package. Document that a full rerun needs a GPU; the
archived package is the evidence for the paper.

## Result-to-evidence map

| Paper item | Source | Verification |
| --- | --- | --- |
| 360 runs, 162 BM25 / 114 dense / 84 RRF | `run_index.jsonl` | `code/audit_verify.py` |
| Table 2 word curves | `runs.jsonl`, `runs.jsonl.gz` | recomputed article means, delta < 0.0001 |
| Table 3 token curves | same | recomputed, delta < 0.0001 |
| Table 4 English control | same | recomputed from `squad_en` dense runs |
| Table 5 best systems | `results/analysis/headline.json`, `summary.json` | `analyze_full.py` with seed 0 |
| Figure 1--4 | `code/make_figures.py` | regenerate from `results/analysis/` |
| 37 N1/N2 failures | `aux_.json`, `manifest.json` | `audit_verify.py` |
| 34 Urdu digits | `manifest.json`, `pool_unicode.txt` | `count_urdu_digits.py` |

The audit script currently reproduces Tables 2--5 and the run counts. It does not reproduce the
figures or the bootstrap endpoints exactly; run `analyze_full.py` and `make_figures.py` for those.

## What cannot be reproduced without new work

- The lower-size dense grid (E1 in `revised_experiments.md`): no runs exist.
- The native Urdu set: does not exist.
- Human relevance judgments: do not exist.
- A published Urdu retrieval baseline: not reproduced.
- The semantic-chunking pilot: not in the 360-run package; do not present it as a main result.



## Public repository

The staged artifact is `public_artifact/` (80.6 MB, no UQA text). It was built by
`code/build_public_artifact.py`. Check these before pushing:

```bash
cd public_artifact
find . -name "corpus_*.jsonl" -o -name "audit_100_questions.csv" -o -name "examples.txt"
python -c "import csv; print(sum(1 for _ in open('data/external/urdu_chunking_full/question_ids_test.csv'))-1)"
# expect 7362
```

Then create an empty public repository (for example `urdu-chunking-artifact`), choose a code
licence, and push:

```bash
cd public_artifact
git init -b main
git add .
git commit -m "Public artifact for the Urdu chunking study"
git remote add origin git@github.com:devxyasir/urdu-chunking-artifact.git
git push -u origin main
```

After the URL resolves, put it in:

- `paper/paper.tex` and `paper/paper_preprint.tex` (Ethics or a Data Availability line);
- `README.md` and `CITATION.cff`;
- `public_artifact/README.md`.

Do **not** add a code licence to UQA text; the artifact excludes it. A sensible split is MIT or
Apache-2.0 for code and CC BY 4.0 for the paper/figures. Do not claim the repository is public
until the URL resolves in a private browser window.

The artifact's `manifest.json` has both `settings_original` (the archived `strict` label) and
`settings.budget_mode_primary = "crossing"`, plus a `corrections` note. That resolves the label
without hiding the original value.

## Current blockers

1. The repository is private. The paper's "released" claim is false until the public artifact exists.
2. UQA text is in the private repository and must be removed from any public copy.
3. `requirements.txt` lacks the GPU stack; a clean GPU rerun is not documented end-to-end.
4. No code licence has been chosen.
5. The paper's URL points to the private repository; do not publish that URL in a preprint until it
   resolves for readers.

Until 1--4 are resolved, the honest status is "private working repository; public artifact pending,"
not "code released."
