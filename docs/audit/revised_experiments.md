# Revised experimental plan

The existing 360 runs are sufficient for the paper's descriptive claims. They are not sufficient for
one inferential claim: that 50 words is the best dense chunk size rather than the smallest size on
the grid. This plan adds the smallest experiment that can settle it, plus optional work that is
needed only if the paper's scope is widened.

Nothing in this file is a result. The commands below have not been executed on this machine; the
local environment has no GPU and no `sentence-transformers`. The script was syntax-checked only.

## E1 (decisive): dense lower-size grid

**Question.** Is the dense optimum at 50 words, below 50 words, or a boundary artefact of the grid?

**Design.** Fixed-word chunks, zero overlap, N0, test split only. Sizes 10, 25, 40, 50. Same 120 test
articles and 7,362 questions as the main study. Four encoders. This is a nested extension of the
existing 50-word run, not a new corpus.

| Item | Value |
| --- | --- |
| Corpus | `data/external/urdu_chunking_full/corpus_ur.jsonl`, `split == test` |
| Queries | `questions.csv`, `split == test` (7,362) |
| Chunker | `code/chunking.py::chunk_fixed_words(text, size, 0.0)` |
| Sizes | 10, 25, 40, 50 words |
| Overlap | 0 |
| Normalization | N0 |
| Encoders | `BAAI/bge-m3`, `intfloat/multilingual-e5-large-instruct`, `Qwen/Qwen3-Embedding-0.6B`, `sentence-transformers/LaBSE` |
| Query format | bge/LaBSE: raw question; e5/Qwen: documented instruction prefix, same as main runs |
| Max sequence | bge 8192, e5 512, Qwen 32768, LaBSE 256 (model settings used in the archived runs) |
| Similarity | cosine over L2-normalized vectors |
| Ranking | top 100 chunks per question |
| Primary metric | BM-Recall@500 (crossing budget), article mean |
| Secondary | Recall@1/3/5/10, MRR@10, nDCG@10, answer-split rate |
| Statistics | 10,000 article-level bootstrap resamples, seed 0; paired article bootstrap for 50 vs 40, 50 vs 25, 50 vs 10 |
| Runs | 16 (4 encoders x 4 sizes) |

**Command (GPU):**

```bash
pip install sentence-transformers torch
python code/run_small_sizes.py --models bge e5 qwen labse --sizes 10 25 40 50 --device cuda
```

Output: `results/small_sizes/runs.jsonl` with one record per run and per-question BM500/split arrays.
The run IDs are `sm_test_<model>_fixed_words_<size>_0_N0_dense`.

**Runtime.** The 10-word grid produces roughly 100k chunks per encoder; 25/40/50 produce fewer.
On an A100, expect 1--3 h for all 16 runs including encoding and top-100 search; on a T4, 4--8 h.
The 10-word run is the memory peak (about 100k x 1024 floats per encoder). Do not run all four
encoders in one process; the script frees each model and empties the CUDA cache.

**Decision rule (pre-register before running):**

- If 40 or 25 is better than 50 by at least 1.0 point and the paired 95% CI excludes zero, the paper
  must replace "50 is best" with the new argmax and rerun the boundary discussion at that size.
- If 50 remains the argmax and the paired CIs for 40/25/10 are entirely below zero, the paper can say
  "50 is the best tested size and the lower bound of the tested range"; it still cannot say
  "interior optimum".
- If the differences are within +/-1 point and CIs cross zero, the result is a plateau; report it as
  such and do not claim a unique optimum.
- Do not add 30/35/45 after seeing the results unless the pre-registered rule requires a follow-up;
  sequential tuning on the test set would invalidate the comparison.

**What this does not fix.** It does not test native Urdu, generation quality, or other domains. It
does not make the 50-word conclusion universal; it only locates the dense boundary more precisely.

## E2 (optional): complete the LaBSE cells

LaBSE was not run with overlap or recursive-English, so the "every model" overlap and strategy
claims currently mean "every model in the sweep except LaBSE." If those claims are kept, run:

- LaBSE fixed_words 50/100 at overlap 10/25 (4 runs)
- LaBSE recursive_english 50/100 at overlap 0 (2 runs)

Six runs, same corpus/questions, same protocol as E1. If not run, rewrite the claims to name the four
systems and say LaBSE was excluded from those sweeps.

## E3 (needed only for a native claim): native-authored query set

The current study cannot support a claim about native Urdu. A minimal set:

- 150--300 native-authored questions over the same 120 test articles (or a new native corpus).
- Exact answer spans marked by two annotators; adjudicate disagreements.
- At least 50 questions re-checked by a second reader for span correctness and answerability.
- Report inter-annotator agreement (span exact match and answer overlap).
- Freeze the set before running any configuration; no tuning on it.

This is a separate data contribution, not a quick extension. Until it exists, every claim stays
"machine-translated UQA."

## E4 (needed only for a RAG-quality claim): human context audit

Sample 100 questions stratified by retrieval outcome (hit/miss) and size (50/200). Two annotators
rate each top-5 context on a 3-point scale: contains the answer, supports the answer, and is not
misleading. Report agreement and the rate of "supports but does not contain." This is the only way to
move from answer-string containment to context quality. It is not needed for the retrieval-only
claims.

## E5 (optional): translation-variant sensitivity

The corpus keeps the modal normalized translation per paragraph (20% of rows). Rebuild the corpus
with a different deterministic rule (first row per paragraph, or a hash-selected row) and rerun the
50/100/200 fixed-word cells for bge only. If the ordering is unchanged, the translation-selection
rule is not driving the result. If it changes, the paper must report the sensitivity. This is cheaper
than E1 but still requires dense encoding; run it only if a reviewer questions the corpus
construction.

## E6 (optional): BM25 lower sizes

BM25 at 10/25/40 is cheap on CPU relative to dense, but it does not answer the dense boundary
question. It is useful only as a descriptive check of whether the sparse curve also turns below 50.
Do not let a BM25 result stand in for the dense experiment.

## What not to run

- Do not add configurations solely to increase the run count.
- Do not tune overlap or size on the test split.
- Do not add a generator and call the result a RAG study without a human or task-level evaluation.
- Do not compare to a published Urdu baseline unless the baseline's exact corpus and tokenizer are
  reproduced; a number copied from a different setup is not a comparison.

## Acceptance criteria for the revised paper

The paper can claim a dense boundary result only when E1 is complete and the decision rule is
applied. Until then, the abstract, results, and conclusion must say **best tested size**, not
**optimum**.
