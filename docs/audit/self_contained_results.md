# Self-containedness audit: result

Three independent raters labeled the same 100 questions. Labels: `1` self-contained, `0`
context-dependent, `?` unclear. Source files are private (`data/audit/reader_A.csv`, `_B`, `_C`);
they contain UQA text and must not be published.

| Reader | 1 | 0 | ? |
| --- | --- | --- | --- |
| A | 74 | 21 | 5 |
| B | 77 | 20 | 3 |
| C | 75 | 21 | 4 |

## Agreement

| Pair | Raw agreement | Cohen's kappa |
| --- | --- | --- |
| A–B | 90.0% | 0.741 |
| A–C | 88.0% | 0.699 |
| B–C | 91.0% | 0.763 |

- Unanimous: **85/100** rows.
- Fleiss' kappa (3 raters, 3 categories): **0.734**.

## Majority label

| Label | Count |
| --- | --- |
| 1 self-contained | 77 |
| 0 context-dependent | 21 |
| ? unclear | 1 |
| no majority | 1 |

Majority self-contained rate: **77/100 = 77.0%**, Wilson 95% CI **67.8–84.2%**.

The one row with no majority was not adjudicated by a fourth reader; it is reported, not silently
assigned. The audit is a corpus check, not a measure of retrieval quality.
