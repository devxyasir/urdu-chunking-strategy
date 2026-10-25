# Self-containedness audit: three readers

100 questions, three raters, labels `1` self-contained, `0` context-dependent, `?` unclear.
The raters worked independently on the same qid order.

| Reader | self-contained (1) | context-dependent (0) | unclear (?) |
| --- | --- | --- | --- |
| A | 74 | 21 | 5 |
| B | 77 | 20 | 3 |
| C | 75 | 21 | 4 |

## Agreement

| Pair | Raw agreement | Cohen's kappa |
| --- | --- | --- |
| A-B | 90.0% | 0.741 |
| A-C | 88.0% | 0.699 |
| B-C | 91.0% | 0.763 |

- Unanimous rows: **85/100**.
- Fleiss' kappa (3 raters, 3 categories): **0.734** (Pbar=0.897, Pe=0.612).

## Majority label

| Label | Count |
| --- | --- |
| 1 | 77 |
| 0 | 21 |
| ? | 1 |
| tie | 1 |

Majority self-contained: **77/100 = 77.0%**, Wilson 95% CI **67.8%--84.2%**.

A row is a tie only when all three labels differ; no tie was adjudicated by a fourth reader.

Do not include the source CSVs in the public repository; they contain UQA text.
