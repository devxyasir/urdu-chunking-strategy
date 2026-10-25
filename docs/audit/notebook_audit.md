# Notebook audit (targeted)

Scope: the cells that produce the 360-run package and the paper's tables. This is not a claim that
every helper was proved correct; it is a record of what was checked against the archived output. The
archived `runs.jsonl`, `run_index.jsonl`, and manifest remain the authority.

## Checked and consistent

- **Config**: four encoders, five chunkers, BM25 whitespace/character, RRF. `MODEL_SPECS` matches the
  paper's model IDs and maximum lengths (bge 8,192; e5 512; Qwen3 32,768; LaBSE 256).
- **Grids**: 21 dense configs (9 fixed-word, 3 fixed-token, 9 boundary), 12 extra 400/800 for bge and
  Qwen3, 6 LaBSE, 21 English, 84 RRF, 162 BM25. This reproduces the 360 count.
- **Splits**: articles are split, not questions. The test questions are 7,362; dev 1,715.
- **Metrics**: `bm500x` is crossing (the chunk that reaches the budget is included); `bm500` is
  strict. `split` is `nrel == 0`, i.e. no chunk contains the full answer. This matches the paper.
- **Bootstrap**: article-level, 10,000 resamples, seed 0; paired contrasts reuse the resample.
- **Normalization**: N1/N2 map spans and drop the 37 whitespace-padded failures; N0 is complete.
- **Truncation**: the notebook sets `truncated = rate > 0.05`. That is why the 3/9,800 LaBSE case
  (0.03%) has `truncated: false` but a nonzero `truncation_rate`. The paper now reports the rate, not
  the boolean. This was a real metadata trap; the rate field caught it.
- **RRF**: fuses cached top-100 BM25 and dense rankings with k=60; the stage checks chunk-set equality
  before emitting.
- **Prefix checks**: 100 dev questions, seed 0, at 200 words; the documented prefix is used for all
  reported runs regardless of the check result.

## Metadata bugs found and handled

1. `write_manifest()` hard-coded `budget_mode_primary="strict"` while the primary metric is
   `bm500x` (crossing). The notebook is patched to write `crossing`; the archived manifest is
   preserved with its original value, and the public copy has `settings_original` + `corrections`.
2. The `truncated` boolean threshold (5%) hid a 0.03% truncation rate. The paper now uses the rate;
   no dense conclusion changes.

## Not audited line by line

- The exact tokenization/normalization helper functions (`normalize_with_map`, `N2_REPL`, BM25
  punctuation stripping) were checked against the manifest and spot examples, not proved line by
  line. The package is the authority.
- The native stage (`stage_F`) was not run; no native results are claimed.
- The notebook was not re-executed from scratch on this machine. The archival package was
  re-aggregated and table-checked instead.

## Practical conclusion

The notebook is suitable as a reproducibility aid, not as a second source of truth. If a reviewer
asks, say: "The archived run package is the evidence; the notebook is the recipe. We audited the
code paths that generate the counts and tables and fixed two metadata bugs; we did not re-run the
full GPU grid." That is accurate and defensible.
