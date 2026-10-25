# Self-containedness audit: instructions

**What it measures.** Whether a question can be understood and answered without reading the article
context. This is a property of the question, not of retrieval quality.

**File.** `data/external/urdu_chunking_full/audit_100_questions.csv` (private; do not put it in the
public repo). Columns: `qid,article_title,question,gold_answer,self_contained`.

**What to fill.** One value per row in `self_contained`:

- `1` = self-contained: the question names its subject/entity well enough to be understood alone.
- `0` = context-dependent: the question relies on the article (pronouns, ellipsis, "the company",
  "he", "this", an unresolved entity, etc.).
- `?` = genuinely unclear; use sparingly and note why in a new `notes` column if you add one.

**Procedure (about 30--45 minutes):**

1. Judge the question **before** reading the answer. Do not open the article.
2. If the answer is needed to decide, judge again after seeing it; record the second judgment.
3. Have at least one additional reader independently fill the same 100 rows (blank the first column before sending).
4. Compare. Record pairwise agreement and majority labels; adjudicate ties separately.
5. Do not change the questions or the sample after looking at the retrieval results.

**Report.**

- Count and percentage self-contained; Wilson 95% CI.
- Disagreement rate before adjudication.
- A short list of ambiguous examples (qids only; no need to quote text in the paper).
- If the context-dependent rate is high, say that retrieval metrics measure answer containment, not
  whether the question is understandable without context. Do not use this audit to "prove" retrieval
  quality either way.

**Run the scorer after filling:**

```bash
python code/score_self_contained.py
```

It checks that all 100 rows are filled, reports the rate and Wilson interval, and writes a small
summary. Do not put the filled CSV in the public artifact; it contains UQA text.
