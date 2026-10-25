# Self-containedness audit: result

File: `data/external/urdu_chunking_full/audit_100_questions.csv` (private; contains UQA text).

| Label | Count |
| --- | --- |
| 1, self-contained | 74 |
| 0, context-dependent | 21 |
| ?, unclear | 5 |
| Total | 100 |

Self-contained rate: **74/100 = 74.0%**, Wilson 95% CI **64.6–81.6%**.

Provenance: one rater, 100 sampled questions, filled after the retrieval runs were complete. This is
a single-rater audit; no second-reader agreement is available. The labels were read from the filled
CSV and scored with `code/score_self_contained.py`.

Interpretation: about one in five audited questions depends on the article context. The corpus is
usable for retrieval evaluation, but a reader should not assume that every question is understandable
without context. Do not treat the audit as a measure of retrieval quality.
