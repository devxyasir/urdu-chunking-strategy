# Pre-registration — How should Urdu documents be chunked for retrieval?

Version 1.0, frozen at GATE 1. Date: 24 October 2026.
Author of record: Muhammad Yasir. Analysis and code: evolve-agent.
This file is written **before any test-split run**. Its SHA-256 is recorded in
`results/phase1/prereg_hash.txt`; the git commit that adds it is the freeze point.

Framing chosen at GATE 0: **option B** — RQ2 and RQ3 are the headline, RQ1 supplies the map,
RQ4 is a parallel-language check, RQ5 is a secondary transfer check. RQ5 is explicitly
underpowered and will be reported as directional only.

---

## 1. Questions and hypotheses

**RQ1.** For Urdu, how do chunk size, overlap and chunking strategy affect retrieval of the
answer-bearing text, for BM25, dense encoders and BM25+dense RRF?

**RQ2.** Is the optimal chunk size stable in **words** across embedding models whose tokenizers
split Urdu differently, and is it more stable in words than in model tokens?

**RQ3.** Do English-toolkit defaults (recursive separators `\n\n`, `\n`, `" "`, `"."` sentence rule)
and English-style text normalization hurt compared with Urdu-aware splitting (`۔ ؟ !`) and
Urdu-aware normalization?

**RQ4.** Is the Urdu optimum different from the English optimum on the parallel SQuAD/UQA data?

**RQ5.** Do chunking conclusions from machine-translated UQA hold on native Urdu text?

| ID | Prediction | Direction | Primary test |
| --- | --- | --- | --- |
| H1 | Chunk-size effects are non-monotonic and the BM25 optimum differs from the dense optimum | two-sided | size x retriever interaction, BM-Recall@500 |
| H2 | The best size is more stable in words than in model tokens across encoders | one-sided (stability) | argmax agreement + curve rank correlation, word vs token arm |
| H3 | Urdu-aware splitting beats English default separators | two-sided | recursive-Urdu vs recursive-English, paired by question |
| H4 | The Urdu and English optima differ on the parallel data | two-sided | argmax difference, Urdu vs English, same model |
| H5 | Normalization matters more for BM25 than for dense | interaction | delta BM-Recall by normalization, BM25 vs dense |
| H6 | UQA config rankings correlate with native-set rankings | one-sided (rho > 0) | Spearman over 36 configs |

H2 is stated as a stability claim, not "words are better". If both units are equally stable,
that is reported as a null. H3 is two-sided because the only published precedent (Khmer,
`2026.2605.22203`) found language-aware chunking *worse*.

## 2. Data

### 2.1 Parallel corpus (UQA + SQuAD2.0)

Source: `uqa/UQA` (Hugging Face), train split 124,745 rows, validation 16,824 rows; English
source: SQuAD2.0 from the UQA repository (`SQuAD/train-v2.0.json`, `SQuAD/dev-v2.0.json`).

**Known property (measured, not assumed):** each English paragraph was translated several times,
producing 4.86 distinct Urdu variants per paragraph on average (max 18). Rows are per question,
and a row's `context` is that translation variant. Declared construction rule:

1. Map every UQA row to its English paragraph by question id (0 unmatched).
2. For each English paragraph, choose the **modal** Urdu variant (ties: longest, then
   lexicographic).
3. A document is the concatenation of the chosen paragraphs of one article, in original order.
4. A question is **retained** only if its row's context equals the chosen variant for its
   paragraph **and** the question is answerable. Gold span = that row's `answer` at `answer_start`
   in the chosen context (verified exact, see below).
5. The English document is the concatenation of the original paragraphs; English gold spans come
   from SQuAD2.0 offsets.

Measured consequence: 24,898 of 124,745 train rows are retained (19.96%).

**Span audit (already run on all rows, not a 100-item sample):** every answerable row's answer
text matches `context[answer_start:answer_start+len(answer)]` exactly — 83,018/83,018 train and
11,169/11,169 validation. The brief's concern about broken MT spans does not materialise in UQA;
this is reported as a result, not hidden.

**Article pool:** 150 English/Urdu article pairs, selected deterministically by SHA-1 of the
English title from the 403 articles with >=20 retained questions, >=20 paragraphs and >=1500 Urdu
words. Split by article (not by question): the 30 articles with `i % 5 == 0` in hash order are
**dev**, the other 120 are **test**. Measured: 1,007,122 Urdu words, 6,720 paragraphs, 9,077
retained answerable questions (dev 1,715 / test 7,362).

### 2.2 Native Urdu corpus

60 long Urdu Wikipedia articles, selected from the 500 most-viewed Urdu Wikipedia articles of
September 2026, filtered to >=1000 words and >=5 paragraphs, then hash-ordered and frozen by
revision ID. Measured: 414,309 words. Text is pinned in `data/processed/native_articles.json`
and readable in `docs/annotation/native_reader.html`.

The user authors **300 questions** (5 per article), each with an exact answer substring; a second
reader checks a random 50. No LLM produces questions, answers or labels. The validator
(`code/validate_native.py`) computes character offsets and rejects ambiguous or absent spans.
Target: >=250 accepted items; if fewer, the native set is reported with its true size.

Native articles are **not** split into dev/test; they are a separate check, never used for
config selection.

### 2.3 Self-containedness audit

Before main runs, 100 UQA questions (stratified by article) are read by the user and rated
self-contained or not. Reported as a rate. If >20% are not self-contained, all claims are stated
as "retrieval of the answer-bearing paragraph", and the affected items are listed.

### 2.4 Licences

- SQuAD2.0: CC BY-SA 4.0.
- UQA: no explicit licence file in the repository or dataset card at the time of writing
  (checked 24 Oct 2026). Treated as research-use; **only code, configs and item IDs are released**,
  not UQA text. A licence inquiry is sent before release.
- Urdu Wikipedia: CC BY-SA 4.0, with attribution and revision IDs.
- Model weights: bge-m3 (MIT), multilingual-e5 (MIT), Qwen3-Embedding-0.6B (Apache-2.0), LaBSE
  (Apache-2.0). Verified at download in Phase 2.

## 3. Design

### 3.1 Factors

- **Corpus:** Urdu UQA (primary), English SQuAD (control), native Urdu (check).
- **Retriever:** BM25-whitespace; BM25-char (3-5 grams); dense (3 encoders in the main grid:
  bge-m3, multilingual-e5-large-instruct, Qwen3-Embedding-0.6B; LaBSE secondary; mpnet excluded for
  its 128-token limit); BM25+dense RRF (k=60).
- **Chunking strategy:** `fixed_words`, `fixed_tokens`, `sentence_aware`, `recursive_english`,
  `recursive_urdu`.
- **Size:** words 50/100/200 (primary, all models); 400/800 (extended, long-context models only).
  Tokens 64/128/256 (token arm, per model tokenizer).
- **Overlap:** 0/10/25% (fixed_words); 0/10/25% for BM25 across all strategies; dense non-fixed
  strategies at 0% only (cost).
- **Normalization:** N0 none, N1 light, N2 full (defined in `design.md` from measured Unicode
  statistics). Tested on BM25 fully; on dense at the best BM25 configuration only.

### 3.2 Grid sizes

| Arm | Models | Configs |
| --- | --- | --- |
| Urdu BM25 | 2 sparse variants | 5 strategies x 5 sizes x 3 overlaps = 75, x 3 normalizations |
| Urdu dense primary | bge-m3, e5-large-instruct, qwen3-0.6b | 21 each (9 fixed-word, 3 token, 3 sentence, 3+3 recursive) |
| Urdu dense extended | bge-m3, qwen3-0.6b | 8 each (400/800, 2 overlaps, 3 strategies) |
| Urdu LaBSE (secondary) | LaBSE | 50/100 words only, 6 configs |
| English control | bge-m3 | 21 |
| Native check | bge-m3, qwen3-0.6b | 21 each |
| Hybrid RRF | all dense configs + BM25 | no extra embedding |

### 3.3 Truncation rule (pre-declared)

Measured: at 200 words, no truncation for bge-m3/e5/qwen; LaBSE truncates 33.9%; at 400 words e5
truncates 74.2%; at 800 words e5 100% and LaBSE 100%. Rule: a config whose chunk-level truncation
rate exceeds **5%** is marked `truncated` and excluded from the primary pooled comparisons; its
raw scores are still reported in the appendix. LaBSE is therefore secondary and limited to 50/100
words. mpnet-multilingual is excluded entirely (its config truncates at 128 tokens).

## 4. Outcomes

**Primary endpoint:** budget-matched recall at **B = 500 words** (BM-Recall@500) for the
answer-bearing text: take chunks in rank order until B whitespace words are reached; success iff
at least one retrieved chunk **contains the full gold answer span**. A span split across two
(possibly overlapping) chunks counts as a failure and is reported in the answer-split rate. One
value per question.

Secondary: BM-Recall@200/1000/2000; Recall@1/3/5/10; MRR@10; nDCG@10; answer-split rate; truncation
rate; words retrieved; chunks retrieved; index size; embedding time.

A chunk is relevant iff it contains the full answer span. Partial spans count as failures and are
reported separately as the answer-split rate.

## 5. Analysis plan

### 5.1 Unit of analysis

Primary unit: **article**, because all questions in an article share the same chunk set and
document. For each config, compute the per-article mean BM-Recall@500, then average across the 120
test articles. CIs: 10,000-resample bootstrap over articles. Question-level means are reported
descriptively; they are not used for inference (they overstate precision by ~2x in simulation).

### 5.2 Pre-declared contrast families

1. **Size effect (RQ1):** for each retriever x strategy, compare adjacent sizes (50 vs 100,
   100 vs 200) and best vs worst. Family of 2 x 5 strategies x 4 retrievers (BM25-whitespace,
   BM25-char, dense, hybrid RRF).
2. **Overlap (RQ1):** 0 vs 10 vs 25% at each size, fixed_words only. 3 sizes x 4 retrievers.
3. **Strategy (RQ1/RQ3):** recursive_urdu vs recursive_english at each size, and sentence_aware vs
   fixed_words. 3 sizes x 4 retrievers.
4. **Unit (RQ2):** word-argmax vs token-argmax per model; cross-model argmax agreement within each
   unit. 4 models.
5. **Normalization (RQ3/H5):** N0 vs N1 vs N2, BM25 vs dense, at the BM25-best config per corpus.
6. **Language (RQ4):** Urdu vs English argmax per model/strategy.
7. **Transfer (RQ5/H6):** rank correlation over the 36 shared configurations.

Holm correction within each family, alpha = 0.05 family-wise. Effect sizes with 95% CIs; every
comparison reports the paired mean difference in recall points and the number of articles.

### 5.3 H2 operationalisation (frozen)

For each encoder m and unit u in {words, tokens}:

- `s_m,u` = the size level with the highest mean BM-Recall@500 on the test split.
- Agreement within a unit = 1 if all main models (bge-m3, e5, qwen) share the same argmax level,
  else 0; report the exact argmax vector.
- Stability is **declared supported** iff (a) word-unit agreement = 1 and token-unit agreement = 0,
  or (b) the cross-model Spearman correlation of the size-response curve is higher in words than in
  tokens by >= 0.2. Otherwise H2 is not supported.
- Equivalence margin: two optima are "the same" if they differ by less than one size step.

Because the primary word grid has only three levels, argmax ties are broken by (i) higher recall,
(ii) smaller size. Ties are reported.

### 5.4 H4 operationalisation

Same model (bge-m3), same strategy, Urdu vs English. H4 supported iff the argmax differs by >= 1
size step and the paired difference at the two optima is significant after Holm. If the optima are
equal, H4 is not supported.

### 5.5 H6 operationalisation

21 configurations per model are shared between UQA and native (bge-m3 and qwen3-0.6b; 42
model-config pairs in total). Spearman rho is computed per model and on the pooled 42 pairs, with a
10,000-resample article bootstrap CI. Directional claim only: rho > 0 with CI excluding 0. A wide
CI is reported as such; no claim of "conclusions transfer". With 21 points the CI for rho = 0.7 is
roughly 0.37-0.88 (Fisher z), so this is a directional check, not an estimate.

## 6. Stopping rules and cut order

- Phase 4 go/no-go: BM25 must beat random by a large margin and the oracle must be 1.0; otherwise
  stop and debug.
- If measured embedding throughput is below 2,500 content tokens/s, apply in order: (1) drop API
  embedders (already dropped), (2) drop the 10% overlap level on dense, (3) drop the 400/800
  extended grid, (4) drop LaBSE, (5) reduce the pool to 100 articles.
- If the native set has <250 accepted items by 25 Nov, RQ5/H6 is reported as a 60-article/≤250-item
  directional check and the paper says so.
- No test-split numbers are inspected before the Phase 5 freeze. Dev-split numbers may inform only
  the pre-declared cut order, never the hypotheses.

## 7. Deviations

Any change after this freeze is recorded in `docs/deviations.md` with date, reason, and whether it
was made before or after seeing test results. Analysis code is committed before the test run and
its hash recorded.

## 8. What this pre-registration does not cover

- The 4B Qwen model (never added; not in the budget).
- Urdu MS MARCO (optional, only after main results; not part of this prereg).
- Semantic/LLM chunking (out of scope; no LLM inside the experiments).
- Any claim about downstream generation quality: no generator is used.
