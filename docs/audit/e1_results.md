# E1 result: bge-m3 lower-size check

**Status:** partial. Four bge-m3 runs completed on the test split at 10/25/40/50 words, zero overlap,
N0. e5, Qwen3, and LaBSE were not extended. The raw per-question records were not added to this
workspace; the table below is the run summary supplied by the author.

| Size | Chunks | BM500x | 95% CI | BM500 (strict) | Answer split |
| --- | --- | --- | --- | --- | --- |
| 10 | 80,962 | 0.4185 | [0.4005, 0.4373] | 0.4167 | 18.23% |
| 25 | 32,423 | 0.6895 | [0.6724, 0.7072] | 0.6861 | 7.73% |
| 40 | 20,285 | 0.7561 | [0.7407, 0.7711] | 0.7528 | 4.82% |
| 50 | 16,244 | 0.7733 | [0.7578, 0.7888] | 0.7677 | 4.10% |

Source: `results/analysis/table_lower.csv`, copied from the author's pasted run summary.

## Reading

- The bge-m3 curve rises monotonically from 10 to 50; 50 remains the best tested size.
- 50 is 1.72 points above 40. The intervals overlap; do not claim a sharp 40-vs-50 separation.
- 10 words is not viable: 18.2% of answers are split across chunks, against 4.1% at 50.
- This is one encoder. The paper must not say "all four encoders were extended below 50."

## What is still needed for a complete E1

- e5, Qwen3, and LaBSE at 10/25/40/50, same protocol.
- Raw per-question arrays for all 16 runs, with run metadata and model revisions, so the paired
  bootstrap can be recomputed.
- If 40 or 25 beats 50 for any encoder with a paired CI excluding zero, update the paper's argmax.
  If not, keep "50 is the best tested size" and state the lower bound.

The notebook is `notebooks/urdu_chunking_small_sizes.ipynb`; the resumable runner is
`code/run_small_sizes.py`.
