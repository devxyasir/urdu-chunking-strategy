# Urdu chunking: public artifact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23648556.svg)](https://doi.org/10.5281/zenodo.23648556)

Public artifact for the paper **"How Should Urdu Documents Be Chunked for Retrieval? A Budget-Matched Study of Chunk Size, Overlap, Sentence Boundaries and Tokenization"**.

## The paper

| | |
|---|---|
| **Title** | How Should Urdu Documents Be Chunked for Retrieval? A Budget-Matched Study of Chunk Size, Overlap, Sentence Boundaries and Tokenization |
| **Author** | Muhammad Yasir ([ORCID 0009-0001-0375-1016](https://orcid.org/0009-0001-0375-1016)) |
| **Status** | Preprint |
| **Zenodo record** | <https://zenodo.org/records/23648556> |
| **DOI** | [10.5281/zenodo.23648556](https://doi.org/10.5281/zenodo.23648556) |
| **PDF** | <https://devxyasir.com/uploads/papers/cfea9039f8131ec6.pdf> |
| **Code and data (this repository)** | <https://github.com/devxyasir/urdu-chunking-strategy> |

### Summary

Most chunking evidence comes from English and measures chunk size in model tokens. This study tests Urdu retrieval on a parallel corpus of 150 articles and 7,362 questions. It compares five chunkers, four multilingual encoders, BM25, and BM25+dense RRF across 360 runs, plus a bge-m3 check at 10 to 50 words.

Main results:

- Among the sizes tested, 50 words is best for every dense encoder. Extending bge-m3 downward is monotonic and leaves 50 words best.
- The best size in tokens differs between encoders because tokenizer fertility varies by about 2.3x.
- Urdu-aware sentence splitting helps at 50 words and reverses at 200 words.
- 25% overlap helps on this corpus. Text normalization makes no difference.
- Three readers judge 77% of sampled questions self-contained.

Limits: the study is retrieval-only and the corpus is machine-translated.

### Cite

Muhammad Yasir. *How Should Urdu Documents Be Chunked for Retrieval? A Budget-Matched Study of Chunk Size, Overlap, Sentence Boundaries and Tokenization.* Zenodo. https://doi.org/10.5281/zenodo.23648556

```bibtex
@misc{yasir_urdu_chunking,
  author    = {Yasir, Muhammad},
  title     = {How Should Urdu Documents Be Chunked for Retrieval? A Budget-Matched Study of Chunk Size, Overlap, Sentence Boundaries and Tokenization},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23648556},
  url       = {https://doi.org/10.5281/zenodo.23648556}
}
```

## About this repository

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

## Run the missing E1 experiment (Colab)

[Open the notebook in Colab](https://colab.research.google.com/github/devxyasir/urdu-chunking-strategy/blob/main/notebooks/urdu_chunking_small_sizes.ipynb).
Upload only `corpus_ur.jsonl` and `questions.csv` to
`MyDrive/urdu_chunking/urdu_chunking_full/`; the notebook clones the code from this repository.
The script is resumable.

For the remaining e5/Qwen3/LaBSE runs (bge-m3 already done), use
[`urdu_chunking_lower_sizes_remaining.ipynb`](https://colab.research.google.com/github/devxyasir/urdu-chunking-strategy/blob/main/notebooks/urdu_chunking_lower_sizes_remaining.ipynb).
It saves `runs.jsonl`, `summary.json`, and a zip to Drive.

## Reproduce the tables (no UQA text needed)

```bash
pip install -r requirements.txt
gunzip -c data/external/urdu_chunking_full/runs.jsonl.gz \
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
