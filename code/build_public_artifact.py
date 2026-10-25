"""Build a public artifact with no UQA text.

The source workspace is private because it contains UQA-derived corpora and question text.
This script copies only code, metadata, per-question metrics, figures, paper PDFs, and
sanitized question IDs into `public_artifact/`. It never copies corpus_ur.jsonl,
corpus_en.jsonl, questions.csv with text, examples.txt, or audit_100_questions.csv.

Usage:
    python code/build_public_artifact.py
Then create a new public repository, copy `public_artifact/` contents to it, and push.
"""
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "external" / "urdu_chunking_full"
OUT = ROOT / "public_artifact"


def copy_file(rel, dst_rel=None):
    s = ROOT / rel
    if not s.exists():
        print("missing", rel)
        return
    d = OUT / (dst_rel or rel)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)


def copy_tree(rel, dst_rel=None, suffixes=None):
    s = ROOT / rel
    if not s.exists():
        print("missing", rel)
        return
    for p in s.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts:
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        r = p.relative_to(s)
        d = OUT / (dst_rel or rel) / r
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "external" / "urdu_chunking_full").mkdir(parents=True)

    # Code and frozen configuration. No data.
    copy_tree("code", suffixes={".py", ".md", ".txt"})
    # The example-figure script embeds a UQA question/answer quote; the figure is shipped instead.
    if (OUT / "code" / "make_example_figure.py").exists():
        (OUT / "code" / "make_example_figure.py").unlink()
    copy_tree("configs")
    copy_tree("figures", suffixes={".pdf", ".png"})
    copy_tree("notebooks", suffixes={".ipynb"})
    copy_tree("docs/audit")
    copy_tree("results/analysis", suffixes={".json", ".csv"})

    # Paper source and PDFs. The short Figure 1 quotation is part of the paper, not a corpus dump.
    for name in ["paper.tex", "paper_preprint.tex", "refs.bib", "acl.sty", "acl_natbib.bst",
                 "paper.pdf", "paper_preprint.pdf", "urdu_chunking_paper.pdf",
                 "urdu_chunking_paper_anon.pdf", "urdu_example.png", "README.md"]:
        copy_file(f"paper/{name}", f"paper/{name}")

    # Safe result package: no corpus, no question/answer text.
    safe = ["run_index.jsonl", "runs.jsonl.gz", "rankings_B.npz", "rankings_C.npz",
            "rankings_D.npz", "rankings_E.npz", "manifest.json", "aux_.json", "summary.json",
            "tokenizer_fertility.json", "RESULTS_README.md"]
    for name in safe:
        copy_file(f"data/external/urdu_chunking_full/{name}",
                  f"data/external/urdu_chunking_full/{name}")

    # Correct the one stale metadata label, preserving the original.
    mpath = OUT / "data" / "external" / "urdu_chunking_full" / "manifest.json"
    man = json.load(open(mpath, encoding="utf-8"))
    original = dict(man.get("settings", {}))
    man["settings_original"] = original
    man["settings"]["budget_mode_primary"] = "crossing"
    man.setdefault("corrections", []).append(
        "settings.budget_mode_primary was hard-coded as 'strict' in the archived manifest; "
        "the primary metric bm500x is the crossing budget. The original value is kept in "
        "settings_original. See docs/audit/verification_report.md."
    )
    json.dump(man, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # Sanitized question IDs in the exact question order used by the per-question arrays.
    rows = []
    with open(SRC / "questions.csv", encoding="utf-8-sig", newline="") as f:
        for q in csv.DictReader(f):
            rows.append((q["qid"], q["article_id"], q["split"]))
    for name, filt in [("question_ids.csv", None), ("question_ids_test.csv", "test")]:
        with open(OUT / "data" / "external" / "urdu_chunking_full" / name, "w",
                  encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "article_id", "split"])
            for qid, aid, split in rows:
                if filt is None or split == filt:
                    w.writerow([qid, aid, split])

    # A minimal questions.csv (no question or answer text) lets analyze_full.py reproduce tables.
    with open(OUT / "data" / "external" / "urdu_chunking_full" / "questions.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "article_id", "split"])
        for qid, aid, split in rows:
            w.writerow([qid, aid, split])

    copy_file("requirements.txt")
    copy_file("requirements-full.txt")
    copy_file("prereg.md")
    copy_file("design.md")

    (OUT / "README.md").write_text("""# Urdu chunking: public artifact

Retrieval-only study of how Urdu documents should be chunked. This repository contains code,
configurations, run metadata, per-question metrics, rankings, figures, and the paper. It does
**not** contain the UQA corpus or the question/answer files: `corpus_ur.jsonl`, `corpus_en.jsonl`,
question/answer text, and `audit_100_questions.csv` are excluded. The paper's Figure 1 contains a
short illustrative quotation.

Public repository: <https://github.com/devxyasir/urdu-chunking-strategy>

## What is here

- `paper/` — revised named and anonymous PDFs, LaTeX source, references.
- `data/external/urdu_chunking_full/` — `run_index.jsonl`, `runs.jsonl.gz` (per-question
  metrics, no text), rankings, manifest, summary, sanitized question IDs and a minimal
  `questions.csv` (qid, article_id, split only).
- `code/` — corpus builder, chunkers, BM25, dense wrappers, analysis, figures, audit script.
- `figures/`, `docs/audit/`, `notebooks/`.

## Reproduce the tables (no UQA text needed)

```bash
pip install -r requirements.txt
gunzip -c data/external/urdu_chunking_full/runs.jsonl.gz \\
  > data/external/urdu_chunking_full/runs.jsonl
python code/analyze_full.py
python code/analyze_extra.py
python code/make_figures.py
```

The run package is 360 runs; `run_index.jsonl` maps every run to its configuration. The
minimal `questions.csv` preserves only IDs and splits, which is all the analysis needs.

## Reproduce the runs (needs UQA + GPU)

UQA has no explicit licence. Obtain it from the original authors, not from this repository.
SQuAD2.0 is CC BY-SA 4.0. The full data build is in `code/`; the archived GPU notebook is
`notebooks/urdu_chunking_fullrun.ipynb`. The lower-size experiment is
`code/run_small_sizes.py` with `notebooks/urdu_chunking_small_sizes.ipynb`.

## What is not claimed

This is a boundary result: 50 words is the smallest dense size tested. The 10/25/40/50
dense grid is specified but was not run when the paper was compiled. The corpus is
machine-translated, single-domain, and retrieval-only. See `docs/audit/publication_readiness.md`.

## Licence

No code licence has been chosen yet. Do not assume a licence for the code or the paper until
one is added. The paper text and figures are the author's; UQA text is not included.
""", encoding="utf-8")

    (OUT / "MANIFEST.md").write_text("""# Artifact map

| Paper item | File |
| --- | --- |
| 360 runs, counts | `data/external/urdu_chunking_full/run_index.jsonl` |
| Per-question metrics | `data/external/urdu_chunking_full/runs.jsonl.gz` |
| Tables 2--5 | `results/analysis/` after `analyze_full.py` |
| Figures | `figures/` after `make_figures.py` |
| Run count audit | `code/audit_verify.py` |
| Lower-size plan/code | `docs/audit/revised_experiments.md`, `code/run_small_sizes.py` |
| Corrections | `data/external/urdu_chunking_full/manifest.json` (`corrections`, `settings_original`) |

The UQA corpus and question/answer files are not included. `questions.csv` has only `qid,article_id,split`.
""", encoding="utf-8")

    # Checks: these names must not exist anywhere in the artifact.
    banned = ["corpus_ur.jsonl", "corpus_en.jsonl", "audit_100_questions.csv", "examples.txt"]
    found = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.name in banned]
    print("banned files found:", found)
    print("artifact bytes:", sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
