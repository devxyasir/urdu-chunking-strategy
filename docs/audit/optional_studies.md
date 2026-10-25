# Optional studies: native queries and human judgments

These are **not required** for the paper's current retrieval-only claims. They are required only if
you want to claim transfer to native Urdu or context quality for RAG. The paper already discloses
both absences. Decide scope before spending money.

## A. Native Urdu query set (closes the MT limitation)

**Minimum useful size:** 150--300 questions over the same 120 test articles. Fewer than 150 will not
support per-article analysis; more than 300 adds little for a first study.

**Who:** two native Urdu readers who have not seen the paper's results. One can write, one can check;
adjudicate disagreements. Do not use machine translation.

**Protocol:**

1. Sample articles stratified by length and by the existing answer-split rate (short/medium/long).
2. For each sampled article, write a question whose answer is a contiguous span **in one sentence**,
   and mark the exact character span. Avoid questions that need outside knowledge.
3. Record `qid, article_id, question, answer, answer_start, answer_end, writer_id, date`.
4. A second reader checks: (a) the span answers the question, (b) the question is answerable from the
   article, (c) the wording is natural Urdu. Record pass/fail and a short reason.
5. Freeze the file before running any configuration. Split by article, never by question.
6. Report inter-annotator agreement: exact-span agreement, answer overlap, and the pass rate after
   adjudication. Do not report only the final agreed set.

**What it buys:** a defensible sentence: "The MT results replicate on N native-authored questions,"
or an honest negative. It does not fix the single-domain or retrieval-only limits.

**Effort:** roughly 2--4 weeks with two readers, depending on how many questions per hour they can
write and check. Budget for payment; do not assume volunteer labour.

## B. Human context audit (closes the "answer-string only" limitation)

**Sample:** 100 questions stratified by outcome (hit/miss) and size (50/200 words), 5 retrieved
contexts each. Two annotators.

**Rubric per context (3 points):**

- 2 = contains the answer and supports it without misleading context.
- 1 = contains the answer but the surrounding context is misleading or incomplete.
- 0 = does not contain or support the answer.

Also record a binary "would a reader accept this as evidence?" and a one-line reason.

**Report:** mean score by size and outcome, agreement (Cohen's kappa or Krippendorff's alpha), and
the rate of "contains but does not support." The last number is the one a RAG reader cares about.

**Effort:** 1--2 weeks; 1,000 judgments is a few days for two annotators if the interface is simple.
Use a spreadsheet or a small HTML form; randomize order and hide the configuration label.

## C. Published baseline reproduction (only if a reviewer demands external comparison)

There is no published Urdu chunking baseline to copy. A defensible baseline is:

- BM25 with a standard tokenizer and the same 500-word budget, on the same 120 articles/questions.
- Report it as a **new run**, not as a literature number.
- If you use a published system, reproduce its corpus, tokenizer, and preprocessing; otherwise the
  comparison is confounded and should be labelled descriptive.

This is a small CPU experiment, not a reason to delay the paper. The current paper compares internal
configurations and does not claim a literature SOTA.

## D. Generation evaluation (only for a RAG claim)

Do not add a generator unless you can also add a task-level metric and a human or strong automatic
judge. A generator with no evaluation turns a clean retrieval paper into a weak RAG paper. Keep the
retrieval-only scope unless you have the budget for a proper end-to-end study.

## Recommendation

For the current paper: **do none of A--D**. Keep the MT-scoped, retrieval-only claims, add E1 (the
lower-size dense grid), and release the artifact. That is a coherent, honest paper.

If you want a stronger follow-up: do A + B on the same corpus. That would support a second paper on
native Urdu and context quality. Do not bolt them onto the current paper without re-running the
analysis and rewriting the claims.
