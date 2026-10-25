# Design — Urdu chunking study

Companion to `prereg.md` (v1.0). This file specifies what the code must do. Any change after the
Phase 5 freeze is a deviation and must be recorded in `docs/deviations.md`.

---

## 1. Repository layout for the experiment

```
code/
  build_corpus.py        # UQA/SQuAD/native -> documents + questions (frozen)
  normalize.py           # N0/N1/N2 + character index maps
  chunkers.py            # five chunking strategies, char spans
  embed.py               # resumable embedding cache (Colab)
  bm25.py                # whitespace and char-n-gram BM25
  rank.py                # retrieve, fuse, write rankings
  metrics.py             # Recall@k, MRR, nDCG, BM-Recall
  analyze.py             # bootstrap, Holm, tables, figures
  run_grid.py            # config driver, one cell per invocation
notebooks/
  colab_runner.ipynb     # thin wrapper: mount Drive, call the scripts
configs/
  grid_*.json            # frozen grids, committed with the code hash
data/processed/          # documents, chunks (git-ignored)
results/                 # metrics and rankings (raw git-ignored, aggregates committed)
tests/                   # pytest, checks (a)-(g)
```

Every script is idempotent and resumable: it derives an output key from its inputs, writes
`<key>.json`/`.npy`, and skips if the key exists and its sidecar hash matches.

## 2. Corpus construction

Inputs:
- `uqa/UQA` train + validation parquet (`id, title, context, question, is_impossible, answer, answer_start`).
- SQuAD2.0 `train-v2.0.json`, `dev-v2.0.json` from the UQA repository.

Steps (frozen; measured counts in `results/phase1/pool_stats.md`):
1. Build `id -> (title, paragraph_index)` from SQuAD. Assert 0 unmatched UQA ids.
2. Per paragraph, choose the modal Urdu context variant (ties: length desc, string desc).
3. Article document = chosen paragraphs joined by `\n\n`, in original paragraph order.
4. Retain a row iff `is_impossible == False` and its context equals the chosen variant.
5. Gold span = `(answer_start, answer_start + len(answer))` in the chosen context. Assert
   `context[start:end] == answer` for every retained row; abort on any mismatch.
6. English document = original SQuAD paragraphs joined by `\n\n`; English gold spans from SQuAD.
7. Split by article using SHA-1(title) order: every 5th is dev (30), the rest test (120).
8. Write `data/processed/corpus_ur.jsonl` and `corpus_en.jsonl`:
   `{article_id, split, title_en, paragraphs:[{pid, text}], questions:[{qid, text, answer,
   answer_start, answer_end}]}`; `article_id = sha1(title_en)[:12]`.

Native set: `native_articles.json` (60 articles, pinned revision IDs) plus
`native_validated.jsonl` from `code/validate_native.py`. The native split is `native`; it is never
used for config selection.

Leakage guard: an article appears in exactly one split; assert no qid or article crosses splits.

## 3. Normalization (N0/N1/N2) and offset mapping

Measured Unicode profile of UQA (`results/phase1/corpus_stats.md`):
- 0 ZWNJ, 0 ZWJ, 0 tatweel, 0 Urdu/Arabic-Indic digits; 32 Arabic yeh; 0 Arabic kaf; 745 combining
  marks; `۔` 56k, `؟` 81, `,` ASCII 795, Arabic comma 72k.
- Therefore normalization is expected to be a near-no-op on UQA. Native text is measured again in
  Phase 2 and the variant definitions are frozen only if the native profile adds the expected
  ZWNJ/tatweel/variant forms; otherwise this is reported.

Variants (applied to documents, questions and gold answers with the same function):
- **N0**: identity.
- **N1 (light)**: NFC; collapse whitespace runs to one space; strip tatweel (U+0640); remove
  Arabic diacritics U+064B-U+0652, U+0670, U+0653-U+0655; map ZWNJ/ZWJ to U+200C/U+200D kept
  (not removed, because they are orthographic in Urdu).
- **N2 (full)**: N1 plus letter unification (ي→ی, ك→ک, ة→ۃ, ہ→ہ no-op), digit unification
  (۰-۹ and ٠-٩ → 0-9), comma/semicolon unification (،→`,`, ؛→`;`), ASCII `?`→`؟`, and removal of
  ZWNJ/ZWJ. `۔` and `؟` are preserved as sentence boundaries.

Offset mapping: `normalize_with_map(text) -> (norm_text, index_map)` where `index_map[i]` is the
original character index of normalized character `i` (for inserted/replaced characters, the start
of the source span). Gold spans are mapped as `(index_map[start], index_map[end-1]+1)`, then
checked: `norm_text[ns:ne] == normalize(answer)`. If the check fails, the question is marked
`span_mapping_failed` and excluded from that variant only; the rate is reported per variant.

Rationale for not searching for the answer: re-finding a span by substring search can silently
label the wrong occurrence. The contract is mapping + equality check, with exclusion and reporting.

## 4. Chunkers

All chunkers operate on a normalized document and return chunks with `(start_char, end_char)`.
Chunks never cross article boundaries. Whitespace tokenization for word units uses `str.split()`.
Sizes are counts of whitespace words unless stated.

1. **fixed_words(size, overlap)**: step = max(1, round(size * (1 - overlap))). Sliding window over
   the word list, sentence-agnostic. Coverage and exact step tested.
2. **fixed_tokens(budget, overlap)**: greedy word-boundary packing: add words while content tokens
   (tokenizer without special tokens) <= budget; never split a word. Step = max(1,
   floor(k * (1 - overlap))) where k = words in the chunk. For LaBSE (256) and e5 (512) budgets are
   capped at max_seq-2.
3. **sentence_aware(target_words, overlap)**: split at `۔`, `؟`, `!` when followed by whitespace or
   end of paragraph; paragraph boundary is also a boundary. Group sentences until the next would
   exceed target; a single sentence longer than 2x target is split by words. Overlap by whole
   sentences until >= target*overlap words. This is the Urdu-aware *segmentation* baseline.
4. **recursive_english(size, overlap)**: separators in order `["\n\n", "\n", ".", " "]`; split
   recursively, merge adjacent units greedily up to `size` words, then apply word overlap. The `"."`
   rule is applied only when followed by whitespace + uppercase or end (English convention).
5. **recursive_urdu(size, overlap)**: separators `["\n\n", "\n", "۔", "؟", "!", " "]`; same merge
   and overlap logic.

For all: the final chunk list must cover every non-whitespace character exactly once at overlap 0;
at overlap r, consecutive chunks share exactly `round(size*r)` words within the fixed-word rule and
a whole number of sentences/units within the sentence/recursive rules (tested per rule).

## 5. Retrievers

**BM25-whitespace**: tokens = whitespace split after stripping leading/trailing punctuation
(`string.punctuation` plus `۔؟،؛«»()`), lowercasing (no-op for Urdu), dropping empty tokens.
k1=1.2, b=0.75, fixed (no tuning; tuning would be a new experiment). Implemented in NumPy with
sparse postings; identical results to `rank_bm25` BM25Okapi on a 1,000-chunk check (test).

**BM25-char**: 3-5 character n-grams over normalized text with spaces collapsed; same BM25 formula
over n-gram terms. Reported as a sparse variant, not mixed with whitespace counts.

**Dense**: `sentence-transformers` in float16 on GPU. Documented query/passage formats:
- `BAAI/bge-m3`: no prefix; CLS pooling; L2 normalize.
- `intfloat/multilingual-e5-large-instruct`: query
  `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q}`;
  passage = raw text; mean pooling; L2 normalize.
- `Qwen/Qwen3-Embedding-0.6B`: query `Instruct: Given a web search query, retrieve relevant
  passages that answer the query\nQuery: {q}`; passage raw; last-token pooling; L2 normalize.
  The exact instruction is verified against the model card in Phase 2 and recorded in the manifest.
- `sentence-transformers/LaBSE`: no prefix; CLS pooling; L2 normalize (secondary, 50/100 words).
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: excluded (128-token config).

Prefix check (test e): on 300 dev questions with bge-m3 and e5, the documented format must beat no
prefix by >=1 recall point at 200 words; if not, the result is reported and the documented default
is kept (no silent switch).

Embedding cache:
`cache/{model_slug}/{sha256(model_revision|corpus_hash|chunk_key|norm|prefix)[:20]}.npy`, float16,
plus a JSON sidecar with shape, dtype, model revision, timestamp and the exact text hash. Never
re-embed an existing key.

**RRF**: `score(d) = sum_over_lists 1/(60 + rank_l(d))`, lists = BM25 and dense for the same chunk
config; union of candidates; ties by chunk id.

**Truncation**: per chunk, tokenize with the model tokenizer; if `len > max_seq - 2`, mark
`truncated=True`, encode the truncated text, and record it. Config-level truncation rate is the
share of chunks marked.

## 6. Metrics

Given ranked chunk ids for a question and the gold span:
- relevant(c) = gold span fully inside c's char span.
- Recall@k = 1[any relevant in top k]; MRR@10 = 1/rank of first relevant (0 if none in top 10);
  nDCG@10 with binary gains and log2 discount.
- **BM-Recall@B** (primary, B=500): walk the ranking, append chunk word counts (words in the chunk
  text, duplicates counted because that is what a reader would read) until the budget is reached;
  success iff any retrieved chunk is relevant. Sensitivity: unique-word budget (merge char spans,
  dedupe words) reported in the appendix.
- Answer-split rate = share of questions where no chunk contains the full span.
- Words retrieved, chunks retrieved, index size (bytes), embedding wall time, truncation rate.

Edge cases fixed by test (f): budget exactly reached mid-chunk; duplicate chunks; empty question;
answer at a chunk boundary; answer split across overlapping chunks (relevant iff one chunk contains
it whole; a split is a failure and counted in the split rate).

## 7. Statistics

- Per-article means for every config/metric; the analysis unit is the article (120 test, 60 native).
- Bootstrap: 10,000 resamples of article ids with replacement, seeded (20261024 + config index).
- Paired contrasts: same resample used for both configs; difference distribution -> percentile CI.
- Holm within the families listed in `prereg.md` section 5.2; alpha 0.05.
- Effect size: mean paired difference in recall points with 95% CI; also the share of articles
  where the direction agrees.
- Ties in argmax: higher recall, then smaller size; ties reported.
- Multiple seeds for chunk boundaries: 5 random start offsets in [0, step) for fixed chunkers on
  the dev split only; spread reported; the test run uses the pre-declared seed 0.

## 8. Tests (Phase 2 acceptance)

| Test | What it proves |
| --- | --- |
| (a) coverage/overlap | union of char spans = document; exact overlap for each rule |
| (b) 40 hand-made Urdu gold examples | diacritics, ZWNJ, mixed digits, punctuation, numbers |
| (c) offset mapping | all variants map spans and pass the normalized-answer equality check |
| (d) oracle/random | oracle BM-Recall = 1.0; random ~ k/N for k=1,3,5,10 |
| (e) embedding determinism/prefix | same text -> identical vector; documented prefix beats none on dev |
| (f) BM-Recall hand cases | budget mid-chunk, duplicates, split answers |
| (g) five printed examples | question, gold chunk, top-3, readable by the user |

Tests run on CPU locally; (e) runs on the GPU in the Colab pilot. Test output is archived under
`results/phase2/`.

## 9. Commands (exact)

```
python code/build_corpus.py --config configs/corpus_v1.json
python code/normalize.py --variant N0
python code/chunkers.py --corpus ur --variant N0 --strategy fixed_words --size 200 --overlap 0
python code/bm25.py --corpus ur --variant N0 --chunks <chunk_key> --mode whitespace
python code/embed.py --model BAAI/bge-m3 --corpus ur --variant N0 --chunks <chunk_key>
python code/rank.py --retriever dense:bge-m3 --chunks <chunk_key> --split test
python code/metrics.py --rankings <rank_key> --budgets 200,500,1000,2000
python code/analyze.py --family size --split test
pytest tests -q
```

The Colab notebook contains the same calls, with Drive mounted at `/content/drive/MyDrive/urdu-chunk`.

## 10. Frozen artefacts

- `configs/grid_primary.json`: 21 dense configs and 75 BM25 configs.
- `configs/models.json`: model ids, revisions, dims, max_seq, prefixes.
- `results/phase1/prereg_hash.txt`: SHA-256 of `prereg.md` and of the commit that freezes this file.
- Phase 5 writes `results/phase5/freeze.json` with the git hash, code hash and config hash.

## 11. Known limitations built into the design

1. UQA text is MT and re-translated per question; the modal-variant rule discards ~80% of rows and
   may select a systematically different translation. Reported, not hidden.
2. The English control uses the same questions but original SQuAD text; differences therefore mix
   language and translation artefacts. This is the point of RQ4/RQ5, but it limits causal language.
3. UQA paragraphs are short (median 145 Urdu words), so documents are concatenations, not naturally
   long texts; chunk boundaries can cross paragraph joins.
4. BM25 hyperparameters are fixed, not tuned; a tuned BM25 is a different experiment.
5. No generator and no LLM judge: conclusions are about retrieval of evidence, not answer quality.
