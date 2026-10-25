# Literature and novelty assessment

This note separates what prior work established, what this paper tests, what the results show, and
what remains unknown. It is not a new survey; the 48 verified works are in `docs/related_work.md`,
`refs/phase0_arxiv.bib`, and `results/phase0/`. Identifiers were checked in Phase 0; the papers were
read for the claims below. None of the prior numbers were reproduced.

## Comparison table

| Work | Language / domain | Chunk unit and grid | Overlap | Retrievers | Evaluation | Main finding | What it does not test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bhat et al. 2025 | English, six QA datasets | Tokens, 64--1024 | Not the focus | Dense and others | Retrieval / QA | Best size depends on answer type and encoder | Urdu; word units; tokenizer fertility |
| Bennani et al. 2026 | English | Tokens, 50--500 | 0 / 20% | SPLADE + generator | Retrieval + generation | Overlap not helpful in their setting | Urdu; dense multilingual; word units |
| Wang et al. 2024 | English | Tokens, 128--2048 | Varies | Dense / RAG pipeline | End-to-end RAG | Chunk size matters inside a larger pipeline | Urdu; controlled unit comparison |
| Singh 2026 | English regulatory | Parsers x chunkers x embedders, factorial | Varies | Dense | Mixed-effects retrieval | Structure-aware choices interact with embedder | Urdu; word/token equivalence |
| Qu et al. 2024 | English | Semantic vs fixed | Varies | Dense | Retrieval cost/quality | Semantic chunking rarely repays its cost | Urdu; multilingual tokenizers |
| Chhoun 2026 | Khmer | Language-aware vs recursive | Not the focus | Dense | Retrieval | The language-aware chunker did worse than recursive | Urdu; size sweep; word/token |
| Alemneh 2026 | Amharic | Passage-level, no chunk sweep | --- | BM25 + dense | Retrieval | BM25 beats four of five zero-shot dense retrievers | Chunking; Urdu |
| UQA (Arif et al. 2024) | Urdu, MT SQuAD2.0 | Paragraphs as released | --- | --- | QA | Provides parallel Urdu QA with spans | Any chunking comparison |
| Urdu MS MARCO (Butt et al. 2024) | Urdu, translated passages | Passages as released | --- | --- | Retrieval | Provides translated passages | Any chunking comparison |
| DRAG (Ahmad et al. 2026) | Urdu | Dual representation; no size sweep | --- | Dense | Retrieval | Dual-representation retrieval for Urdu | Chunk size, overlap, boundaries |
| CURE (Iqbal et al. 2020) | Urdu | Documents | --- | Sparse | IR collection | Stemming/query expansion resource | Chunking; dense retrieval |
| Qureshi et al. 2021 | Urdu | Indexing/normalization | --- | Sparse | Indexing | Describes normalization rules but does not test retrieval effects | Controlled normalization comparison |
| Petrov et al. 2023; Ahia et al. 2023; Lundin et al. 2025; Srivastava et al. 2026 | Multilingual / Indic | Tokenizer analysis | --- | --- | Tokenization / downstream | Tokenizers treat languages unequally; fertility can predict accuracy | Retrieval chunking under matched word budgets |

The table is deliberately narrow. It omits the many English RAG engineering reports that do not
report a controlled chunking comparison.

## What prior work established

1. Chunk size changes retrieval and downstream QA; there is no universal best size in English.
2. Overlap is not a reliable win; at least two English studies report neutral or negative effects.
3. Semantic and structure-aware chunkers can help in some domains, but the gains are conditional and
   often do not repay their cost.
4. Tokenizers allocate different numbers of tokens to the same text across languages; token-defined
   budgets are therefore not comparable across models.
5. Urdu has QA/retrieval resources (UQA, Urdu MS MARCO, CURE, DRAG) but no published controlled
   chunking evaluation that we found.
6. In at least one low-resource language (Amharic), BM25 remains a strong baseline against zero-shot
   multilingual dense retrievers; multilingual averages hide this.

## What this paper independently tests

1. A **unit comparison**: fixed word sizes and fixed token budgets on the same Urdu corpus, with
   measured fertility for four tokenizers.
2. A **boundary interaction**: Urdu-aware sentence splitting versus the English recursive default at
   50/100/200 words, including the sign reversal at large sizes.
3. **Overlap** at 0/10/25% on a machine-translated, paragraph-concatenated corpus.
4. **Normalization** (N0/N1/N2) with a Unicode audit of what is actually present in the corpus.
5. A **parallel English control** on the same articles and questions (bge-m3 only).

## What the results show

- Among 50/100/200-word chunks, 50 is the best tested size for all four dense encoders; BM25's best
  tested fixed-word size is 200--400. The dense curves fall monotonically from 50, so the result is a
  boundary result, not an interior optimum.
- The numeric token argmaxes differ (128 bge, 64 e5, 256 Qwen3), but adjacent budgets are
  statistically tied. The durable difference is the amount of text a budget buys: about 96 words for
  XLM-R at 128 tokens and 42 for Qwen3.
- Sentence-aware splitting beats the English recursive default by 5.5--6.9 points at 50 words for
  bge/e5/Qwen/BM25, and loses 3--12 points at 200 words (9--12 for the dense encoders).
- 25% overlap is associated with 2.6--7.2 points higher BM-Recall@500 on this corpus, in the
  configurations where it was run.
- Normalization is a null on UQA; the corpus has almost nothing to normalize (no ZWNJ/ZWJ, one
  tatweel, 34 Urdu digit characters).
- English is easier on the parallel control, but the best tested fixed-word size is also 50.

## What remains unknown

- Whether a dense optimum exists below 50 words. This is the most important open question; the
  current grid cannot answer it.
- Whether the findings transfer to native Urdu, web text, legal text, conversation, or long
  documents.
- Whether overlap helps when answers are longer or when the generator, not retrieval, is scored.
- Whether a tuned BM25 or a different fusion weight changes the sparse picture.
- Whether human readers would judge the retrieved chunks as good context; the metric is
  answer-string containment.

## Defensible contribution statement

> We provide a systematic, budget-matched chunking evaluation for Urdu retrieval on a parallel
> UQA--SQuAD2.0 corpus. The study isolates two variables that are usually confounded: the unit of
> chunk size (words versus model tokens) and the boundary rule (English recursive defaults versus
> Urdu-aware sentence splitting). Across four multilingual encoders, the best tested word size is
> the same (50 words), while the numeric token argmax differs because tokenizer fertility differs by
> 2.3x. Sentence-aware splitting helps at 50 words and reverses at 200, and 25% overlap helps on this
> machine-translated corpus. We also report a null result for orthographic normalization and a
> parallel English control. The results are retrieval-only and do not establish an interior optimum
> or a generator-level benefit.

Do **not** claim:

- "first" (we found no prior study, which is weaker than proof of absence);
- "optimal" or "state of the art";
- "generalises to Urdu retrieval" beyond this corpus;
- "overlap helps RAG" (no generator was evaluated);
- "normalization does not matter for Urdu" (it is a null on this MT corpus, not on native text).
