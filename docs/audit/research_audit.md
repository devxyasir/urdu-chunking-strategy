# Research audit

Audit of *How Should Urdu Documents Be Chunked for Retrieval? A Budget-Matched Study of Chunk Size,
Overlap, Sentence Boundaries and Tokenization*. Evidence comes from `runs.jsonl`, `run_index.jsonl`,
`manifest.json`, `results/analysis/`, and the LaTeX source. The verification log is
`docs/audit/verification_report.md`.

## Verdict in one paragraph

The measurements are internally consistent and the main tables reproduce exactly from the
per-question arrays: 360 unique runs, 7,362 test questions, 120 test articles, no duplicate run IDs,
no length or ranking misalignment. The paper is not fabrication. It does, however, contain a real
arithmetic error in the appendix, several claims that are stronger than the grid supports, one false
statement about truncation, and a reproducibility claim that the current private repository does not
satisfy. Those are fixable without new retrieval runs. The one scientific gap that needs new compute
is the untested region below 50 words for dense encoders.

## Fix status in the current revision

| Issue | Status |
| --- | --- |
| C1 appendix 336 vs 360 | Fixed; appendix now itemizes 162 + 114 + 84 = 360 |
| C2 artifact claim | Claim changed to "prepared for release; private during review"; 80.6 MB staging artifact built (`public_artifact/`); public repo not created |
| C3 first claim | Removed from the contribution bullet |
| C4 truncation | Fixed; 3/9,800 LaBSE chunks, 256-token window |
| C5 optimum wording | Fixed to "best tested size" in abstract, results, discussion, conclusion |
| C6 AI disclosure | Broadened |
| M1 lower dense grid | Partly done: bge-m3 10/25/40/50 (50 still best); e5/Qwen3/LaBSE not run |
| M2 LaBSE sweep scope | Claims narrowed to four systems; optional runs not done |
| M15 manifest budget label | Paper appendix notes it; notebook patched for future reruns; archived manifest unchanged |

## Critical issues (must fix before release)

### C1. The appendix experiment count is wrong: 336 listed, 360 actual

The appendix says:

- BM25: 2 variants x 5 strategies x 5 sizes x 3 overlaps = 150.
- Dense: 21 per model for bge, e5, Qwen; 6 LaBSE; 12 extra 400/800; 21 English; 84 RRF.
- Total 360.

Those components sum to **336**. The missing 24 runs are visible in `run_index.jsonl`:

- BM25 has 75 base runs per variant **plus 6 N1/N2 normalization runs per variant** (3 sizes x 2
  normalizations), so 81 + 81 = **162**, not 150.
- The 400/800 extension was run for **both** bge and Qwen, i.e. 12 per model = **24**, not 12.
- Dense is therefore 54 bge + 21 e5 + 33 Qwen + 6 LaBSE = **114**, not 102.

162 + 114 + 84 = **360**. The run inventory is correct; only the appendix arithmetic is wrong. Fix the
appendix, not the run count. The verification report records the full breakdown.

### C2. "Code ... released" is not true of the current repository

The abstract, Ethics section, Conclusion, and README say code, configurations, and per-question
metrics are released. The repository is private and no public artifact link appears in the paper. A
reader cannot verify the claim. This is a reproducibility claim, not a stylistic one. Either:

1. publish a code/results artifact that contains no UQA text (code, configs, IDs, per-question
   metrics, manifest, figures), and put its URL in the paper; or
2. change every "released" sentence to "prepared for release" / "available from the authors during
   review" and say the artifact will be public on publication.

Do not leave the current wording.

### C3. The "first systematic" claim is broader than the evidence

The contribution list says "the first systematic chunking evaluation for Urdu retrieval, to our
knowledge." The Phase 0 search found no such paper, but 48 verified works are not a proof of absence.
The claim adds little; the unit result is the contribution. Replace with "a systematic,
budget-matched chunking evaluation for Urdu retrieval" and, in the introduction, "we found no prior
study that varies Urdu document segmentation." Keep the hedge on the search, not on the field.

### C4. The truncation statement is false

Methods says "No dense run reports a truncated chunk." The run index records **3 truncated chunks
out of 9,800** in `ur_test_labse_sentence_aware_100_0_N0_dense` (0.031%). All other dense runs have
zero. Rewrite: "Only LaBSE's 100-word sentence-aware run truncates: 3 of 9,800 chunks (0.03%) exceed
its 256-token window; the effect on the reported mean is negligible but it is not zero."

### C5. "Optimum" language is inconsistent with a boundary result

The abstract says "Word-defined optima agree"; Results says "At its optimum"; Discussion says "The
shared optimum is the smallest size tested"; the next sentence says "They do not show that 50 is
optimal in any absolute sense." These cannot all be true. Use **best tested size** everywhere for the
word grid, and reserve "optimum" for a tested interior point. The same applies to "English ... same
fixed-word optimum" and "At its optimum bge-m3".

### C6. AI-use disclosure is understated

The Ethics section says "AI assistance was limited to coding and editing." The study also used AI for
literature search, analysis scripting, figure generation, and drafting/editing. Under most venue
policies this is a material disclosure. Replace with a factual statement: "AI assistance was used for
literature search, code development, analysis, figure generation, and drafting/editing. The author
verified all results and takes responsibility for the content." Check the target venue's policy
before submission.

## Major issues

### M1. The dense grid stops at 50 words for three encoders, so Q2 is only half-answered

All four dense encoders are best at 50, the smallest tested size. The paper discloses this and says a
finer grid is future work. That is honest, but the abstract's "word-defined optima agree" still reads
as an optimum. The defensible claim is: *among 50/100/200 words, 50 is best for every dense encoder;
an interior optimum below 50 is untested.* A 10/25/40/50 grid is the single most valuable new
experiment. It is specified in `docs/audit/revised_experiments.md`; it needs GPU time, which this
machine does not have.

### M2. "Every model" claims exclude LaBSE from two sweeps

LaBSE has no overlap runs and no recursive-English runs. The overlap section says "25% overlap helps
at every size and model"; the strategy section says sentence-aware beats the default "in every
system." In both cases the tested systems are bge/e5/Qwen/BM25. Rewrite to "every system in the sweep
(LaBSE was not run with overlap/recursive-English)" or add the missing LaBSE runs. The abstract's
5.5--6.9 range is correct for the four tested systems; do not extend it to LaBSE.

### M3. The token result is a plateau, and the abstract hides the plateau

The numeric argmaxes are 128/64/256, but adjacent budgets are statistically tied (bge 64 vs 128
-0.08 points; e5 +0.35; Qwen 128 vs 256 -0.09). "Token optima do not agree" is technically true of
the argmax but rhetorically stronger than the data. Say "the argmax differs, but adjacent budgets are
tied; the durable difference is how much text a budget buys."

### M4. No native Urdu evaluation

UQA is machine-translated; every query and answer is translated. The paper discloses MT but does not
test whether the findings hold for native questions. This limits external validity, not internal
validity. The 100-question self-containedness audit was exported (`audit_100_questions.csv`) but not
filled, so even the rate of context-dependent questions is unknown. State this explicitly in
Limitations; the native set is a separate study.

### M5. English control confounds language with translation

The Urdu side is MT; the English side is original SQuAD2.0. The 7.4-point gap cannot be assigned to
"Urdu is harder" alone. The paper says this in Limitations, but the Results sentence "English is
easier" and the Discussion's "translation artifacts and weaker Urdu coverage" should be phrased as
two candidate explanations, not a decomposition. Either back-translate the English side or add a
native control; neither is available now.

### M6. One corpus, one domain, one answer type

Wikipedia articles built from short paragraphs; short extractive spans. The overlap result is
explicitly corpus-specific. The conclusion "25% overlap helps on this corpus" is appropriately
narrow, but the abstract and README drop "on this corpus." Add it. Do not generalise to web, legal, or
conversational Urdu.

### M7. UQA licence and redistribution

UQA has no explicit licence. The repository contains derived UQA text (`corpus_ur.jsonl`,
`questions.csv`) and the paper includes an Urdu example. The paper says code/IDs only, but the
repository does not yet separate them. Before any public release: remove UQA text from the public
artifact, keep only IDs and per-question metrics, and add a data-acquisition script. The short
example in the paper is a quotation for criticism; keep it short and attribute it.

### M8. Relevance is answer-string containment, not human judgment

A chunk is relevant if it contains the exact gold answer string. That is reproducible and
appropriate for extractive QA, but it is not a human relevance judgment and does not measure whether
the chunk is a good context. The paper's "retrieval only" limitation covers part of this; add "no
human relevance or answer-quality judgments."

### M9. BM25 is fixed, not tuned; RRF is one k

`k1=1.2`, `b=0.75`, and RRF `k=60` were fixed in advance. That is defensible for a controlled
comparison, but a tuned BM25 could move the sparse curve. State it in Methods (it is in Limitations)
and do not call BM25 "the sparse baseline" without "untuned."

### M10. Multiple comparisons and selection

The paper reports many article-level CIs and one p=0.038 normalization result. The CIs are marginal,
not adjusted for multiplicity, and the best system is selected after looking at all runs, so its CI
is optimistically narrow. The paper mentions multiple comparisons in Limitations; add the
selection-adjusted caveat next to Table 5 and avoid treating p=0.038 as a discovered effect.

### M11. Token arm uses only three budgets and one tokenizer per model

The token result rests on 64/128/256. There is no 32 or 512, and the word extents are averages, not
per-chunk measurements. The claim "128 tokens is about 96 words" is a corpus-average; say "on this
corpus, on average." The 96/42 figures are useful but not universal constants.

### M12. No published Urdu retrieval baseline was reproduced

The study compares internal configurations, not prior reported numbers. That is appropriate because
no prior Urdu chunking benchmark exists, but it means the absolute BM25/dense numbers cannot be
positioned against the literature. Do not add a "state of the art" claim. If a baseline is desired,
reproduce BM25 on UQA with a published tokenizer and report it as a new run, not as a literature
number.

### M13. 37 N1/N2 failures are excluded, not imputed

37 of 9,077 questions (0.41%) fail span mapping after normalization because UQA answers are
whitespace-padded. They are excluded from N1/N2 only; N0 is complete. This is documented, but the
normalization null is computed on a slightly different question set. Report the N0/N1/N2 counts in
the appendix and note that the null is not affected materially; do not hide the exclusion.

### M14. The 4.86-variants-per-paragraph corpus construction selects the modal translation

Keeping the modal normalized translation per row retains 20% of 124,745 rows. That is a reasonable
deduplication rule, but it may select a systematically different translation style. The paper
discloses it. A sensitivity run on a different variant (e.g., first row per paragraph) would test
this; it is optional but cheap relative to dense runs.

### M15. The archived manifest's budget label contradicts the paper

`manifest.json` says `"budget_mode_primary": "strict"`, while the paper's primary metric is the
**crossing** budget `bm500x`. The metric code is unambiguous: `bm500x` includes the chunk that
reaches the budget, and `bm500` is strict. The manifest key is a hard-coded label from the notebook,
not the computation. I patched `notebooks/urdu_chunking_fullrun.ipynb` so a future rerun writes
`"crossing"`, but the archived manifest remains as evidence. The paper now has an appendix note. A
reviewer who opens the manifest will otherwise think the primary metric is strict; keep the note.

## Minor issues

- `refs.bib` has 48 entries, 36 cited. The 12 uncited entries do not appear in the PDF, but they make
  the artifact look padded. Either cite them where relevant or remove them.
- "Urdu has more than 230 million speakers" has no citation. Add a demographic source or soften.
- "None of them varies how documents are segmented" (recent Urdu resources) and "Segmentation has
  not been touched" are absolute. Use "we found no ...".
- "At its optimum bge-m3" -> "At 50 words, bge-m3".
- "A sparse retriever wants a chunk..." -> "BM25's best tested size is..." (avoid anthropomorphism).
- "Every dense encoder does best at 50 words and then declines, with no second peak" — LaBSE was only
  tested at 50/100; say "in the tested range."
- "Against the English literature, 25% overlap helps" — two cited English studies, not a literature;
  say "whereas the English studies we reviewed found neutral or negative effects."
- "At 200 words it loses by 9--12" — this excludes BM25 (-3.09). Say "9--12 for dense encoders,
  3 for BM25."
- Table 2 caption says 200 and 400 are tied; add the paired difference and p-value if space allows.
- "No dense run reports a truncated chunk" (C4).
- "Fixed seeds" — state the bootstrap seed (0) and the number of resamples (10,000); dense encoding is
  deterministic, but say so.
- "Per-question metrics are released" (C2).
- README headline "Word-defined optima agree" repeats C5; update README with the paper.
- The paper's example Urdu sentence is from UQA; keep it short and add "example for illustration;
  not redistributed as corpus."
- The title says "Budget-Matched"; the paper never defines the term in one sentence. Add: "All
  comparisons use the same 500-word reading budget at query time; indexing budgets differ by design
  and are reported as chunk counts and index sizes."
- "Budget-matched" can be confused with equal index size or equal embedding cost. The paper is
  query-budget-matched, not index-budget-matched. State that explicitly.

## What is verified

- 360 unique run IDs; 360 rows in `runs.jsonl`; no duplicates.
- BM25 162 (81 whitespace + 81 character), dense 114 (93 Urdu + 21 English), RRF 84.
- All per-question arrays have 7,362 entries; all ranking files align.
- Tables 2--5 (words, tokens, English control, best systems) reproduce from `runs.jsonl`
  (article means; deltas < 0.0001).
- 120 test articles, 7,362 test questions, 1,007,122 pool words, 150 articles.
- 37 N1/N2 failures; 2 documents not NFC; 34 Urdu digit characters; 0 ZWNJ/ZWJ; 1 tatweel; 9
  Arabic-variant yeh.
- 48 bibliography entries have resolvable identifiers (Phase 0); 36 are cited; no cited key is
  missing from the bib.
- 0.60 GPU-hours of recorded embedding (2,162.4 s), consistent with the Colab session.

## What is not verified

- CIs were regenerated by `analyze_full.py` and match the headline values, but the bootstrap is
  stochastic; exact CI endpoints can differ in the last digit across numpy versions. The reported
  intervals are reproducible with seed 0 and 10,000 resamples.
- The semantic-chunking pilot is not in the 360-run grid; no semantic runs appear in `runs.jsonl`.
  Do not describe it as one of the five main chunkers.
- The 100-question self-containedness audit is unfilled.
- The native Urdu set does not exist.
- No published baseline was reproduced.
- The repository is private; no public artifact was verified.
- I did not re-run dense embeddings; the numbers are taken from the archived runs and re-aggregated.

## Required new experiments

1. Dense fixed-word grid at 10/25/40/50 words, overlap 0, N0, four encoders (the decisive test for
   the boundary). Exact configuration in `docs/audit/revised_experiments.md`.
2. Optional: LaBSE overlap and recursive-English runs to complete the "every model" claims, or
   narrow the claims.
3. Optional: native-authored Urdu query set; without it, all claims stay MT-scoped.
4. Optional: human relevance/answer-quality audit on a stratified sample; not required for the
   retrieval claims but required for any RAG-quality claim.

## Priority order

1. Fix C1 (appendix count), C4 (truncation), C5 (optimum language), C3 (first), C6 (AI disclosure).
2. Fix C2 (artifact) by either releasing or narrowing.
3. Fix M2/M3/M5/M6 wording.
4. Add seed, budget definition, N1/N2 counts, selection caveat.
5. Run or commission the 10/25/40 dense grid before claiming an optimum.
6. Decide whether to run the optional native/human studies; if not, keep the MT-scoped title and
   abstract.
