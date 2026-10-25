# Verification report

- run_index rows: 360
- runs.jsonl rows: 360
- test questions: 7362
- test articles: 120

## 1. Experiment count

- by retriever: {'bm25_ws': 81, 'bm25_char': 81, 'dense': 114, 'rrf': 84}
- by stage: {'B': 162, 'C': 93, 'D': 21, 'E': 84}
- by retriever/model: {('bm25_ws', 'None'): 81, ('bm25_char', 'None'): 81, ('dense', 'BAAI/bge-m3'): 54, ('dense', 'intfloat/multilingual-e5-large-instruct'): 21, ('dense', 'Qwen/Qwen3-Embedding-0.6B'): 33, ('dense', 'sentence-transformers/LaBSE'): 6, ('rrf', 'BAAI/bge-m3'): 30, ('rrf', 'intfloat/multilingual-e5-large-instruct'): 18, ('rrf', 'Qwen/Qwen3-Embedding-0.6B'): 30, ('rrf', 'sentence-transformers/LaBSE'): 6}
- distinct run ids: 360
- duplicate run ids: 0
- BM25 by normalization: {('bm25_ws', 'N0'): 75, ('bm25_char', 'N0'): 75, ('bm25_ws', 'N1'): 3, ('bm25_char', 'N1'): 3, ('bm25_ws', 'N2'): 3, ('bm25_char', 'N2'): 3}
- dense Urdu runs: 93; dense English: 21; dense total: 114
- RRF runs: 84

Paper appendix arithmetic:

- BM25: 2 x 5 x 5 x 3 = 150 (omits 6 normalization runs per variant = 12)
- Dense: 21x3 + 6 + 12 + 21 = 102 (the 12 extended runs are per model for bge and Qwen: 24)
- RRF: 84
- Appendix total: 150 + 102 + 84 = 336
- Actual: BM25 162 + dense 114 + RRF 84 = 360.

## 2. Per-question length and ranking alignment

- runs with wrong per-question length: 0
- rankings (runs, misaligned): {'rankings_B.npz': (162, 0), 'rankings_C.npz': (93, 0), 'rankings_D.npz': (21, 0), 'rankings_E.npz': (84, 0)}

## 3. Paper tables re-derived from per-question arrays

| config | paper | recomputed | delta |
| --- | --- | --- | --- |
| bge-m3 50w | 0.7734 | 0.7734 | -0.0000 |
| bge-m3 100w | 0.7569 | 0.7569 | +0.0000 |
| bge-m3 200w | 0.6796 | 0.6796 | -0.0000 |
| multilingual-e5-large-instruct 50w | 0.7756 | 0.7756 | -0.0000 |
| multilingual-e5-large-instruct 100w | 0.7515 | 0.7515 | +0.0000 |
| Qwen3-Embedding-0.6B 50w | 0.6029 | 0.6029 | -0.0000 |
| Qwen3-Embedding-0.6B 100w | 0.5558 | 0.5558 | -0.0000 |
| LaBSE 50w | 0.5404 | 0.5404 | +0.0000 |
| LaBSE 100w | 0.4866 | 0.4866 | +0.0000 |
| BM25 50w |  | 0.6477 | |
| BM25 100w |  | 0.6842 | |
| BM25 200w |  | 0.7076 | |
| BM25 400w |  | 0.7085 | |
| BM25 800w |  | 0.6123 | |

### Token table

| config | paper | recomputed | delta |
| --- | --- | --- | --- |
| bge-m3 64t | 0.7671 | 0.7671 | +0.0000 |
| bge-m3 128t | 0.7679 | 0.7679 | +0.0000 |
| bge-m3 256t | 0.6773 | 0.6773 | +0.0000 |
| multilingual-e5-large-instruct 64t | 0.7651 | 0.7651 | +0.0000 |
| multilingual-e5-large-instruct 128t | 0.7616 | 0.7616 | -0.0000 |
| Qwen3-Embedding-0.6B 64t | 0.4888 | 0.4888 | +0.0000 |
| Qwen3-Embedding-0.6B 128t | 0.5890 | 0.5890 | -0.0000 |
| Qwen3-Embedding-0.6B 256t | 0.5898 | 0.5898 | +0.0000 |

### English control (bge-m3)

| strategy | size | Urdu (paper) | Urdu recomputed | English (paper) | English recomputed | diff paper | diff recomputed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_words | 50 | 0.7734 | 0.7734 | 0.8474 | 0.8474 | +7.4 | +7.40 |
| fixed_words | 100 | 0.7569 | 0.7569 | 0.8288 | 0.8288 | +7.2 | +7.19 |
| fixed_words | 200 | 0.6796 | 0.6796 | 0.7577 | 0.7577 | +7.8 | +7.81 |
| recursive_english | 50 | 0.8276 | 0.8276 | 0.9351 | 0.9351 | +10.8 | +10.75 |
| recursive_english | 100 | 0.8357 | 0.8357 | 0.8988 | 0.8988 | +6.3 | +6.31 |
| recursive_english | 200 | 0.8092 | 0.8092 | 0.8000 | 0.8000 | -0.9 | -0.92 |

### Best systems

- BAAI/bge-m3: 0.8878
- intfloat/multilingual-e5-large-instruct: 0.8855
- RRF BAAI/bge-m3: 0.8709
- RRF intfloat/multilingual-e5-large-instruct: 0.8652
- RRF Qwen/Qwen3-Embedding-0.6B: 0.8137
- RRF sentence-transformers/LaBSE: 0.8103
- BM25: 0.7489

## 4. Dataset facts

- uqa_rows: 124745
- english_paragraphs: 19035
- paragraphs_covered_by_uqa: 19021
- retained_rows: 24898
- eligible_articles: 403
- pool_articles: 150
- pool_dev: 30
- pool_test: 120
- pool_words: 1007122
- pool_paragraphs: 6720
- pool_questions: 9077
- dev_questions: 1715
- test_questions: 7362
- N1/N2 failures: 37 / 9077
- docs not NFC: 2
- tokenizer fertility: {'bge': {'ur': 1.339574716558805, 'en': 1.4804985679692206}, 'e5': {'ur': 1.339574716558805, 'en': 1.4804985679692206}, 'qwen': {'ur': 3.069894409031191, 'en': 1.3808710188929727}, 'labse': {'ur': 1.24849399056007, 'en': 1.3150836478064731}}

## 5. Citation keys

- cited keys: 36; bib keys: 48
- cited but missing: []
- bib but uncited: ['belinchon2026the', 'faheem-etal-2026-u', 'faisal2024legaluqa', 'fiaz2025urdullama', 'finardi2024the', 'kazi2021uquad10', 'kreileder2026evaluating', 'louis2024colbertxm', 'migielski2026chunking', 'mishra2025longcontext', 'rust2020how', 'zhong2026semantic']

## 6. Run flags

- truncated runs: 0
- runs with truncation_rate > 0: 1
- total recorded embedding seconds: 2162.4

## 7. What this report establishes

- **Verified:** 360 unique runs; 162 BM25 (81 whitespace + 81 character), 114 dense (93 Urdu + 21 English), 84 RRF; all per-question arrays length 7,362; all ranking files aligned; Tables 2--5 reproduce from the archived per-question arrays to within 0.0001.
- **Verified:** 120 test articles, 7,362 test questions, 150-article pool, 1,007,122 words; 37 N1/N2 failures; 2 non-NFC documents; 34 Urdu digit characters.
- **Not verified here:** the bootstrap endpoints in the last digit (stochastic; seed 0, 10,000 resamples); the figures; any dense embedding rerun; the semantic pilot (not in the 360-run grid); the native set (does not exist); human judgments (do not exist); a public artifact (repository is private).
- **Requires a new experiment:** the 10/25/40/50 dense grid. No current run can answer whether 50 is an interior optimum.
