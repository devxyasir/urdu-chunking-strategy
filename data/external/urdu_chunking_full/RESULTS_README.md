# Urdu chunking study: results package

Order of questions: rows of questions.csv with split == the run's split, in file order. Every per_question array and every
row of `top10` follows that order. Native runs use questions_native.csv (all rows).

- manifest.json: schema, timestamps, GPU/library versions, SHA-256 of inputs, model revisions, expected vs actual counts, self-test results, tokenizer fertility, prefix checks, settings, deviations.
- corpus_ur.jsonl / corpus_en.jsonl: one record per article (doc text, paragraphs, questions with doc-level offsets).
- questions.csv: Urdu question/answer/offsets plus English question_en/answer_en/offsets for the English control.
- runs.jsonl: one record per run with metrics and per_question arrays (schema in the task spec). Extras: `bm*x` = crossing-budget variant (last chunk may overshoot), `chunk_words_mean`, `chunk_tokens_mean/p95`, `budget_overflow_rate` (fixed_tokens), `excluded_question_positions` (N1/N2 mapping failures; their per_question entries are -1).
- run_index.jsonl: same as runs.jsonl without per_question (light).
- rankings_<stage>.npz: keys `<run_id>|top10` (int32 [n_questions,10] global chunk ids, -1 = none/excluded) and `<run_id>|spans` (int32 [n_chunks,3] = article_index, start, end in the normalized document of that run). `run_ids` lists all runs. Chunk ids index rows of that run's `spans`. article_index = article_id in corpus_*.jsonl.
- audit_100_questions.csv: 100 dev questions for the human self-containedness audit (fill `self_contained`).
- examples.txt: five full examples for three strategies (chunk-boundary inspection).
- summary.json: the SUMMARY_JSON printed at the end.
- tokenizer_fertility.json: tokens per word (Urdu / English) per model tokenizer.
- state.json: completed run ids.

Definitions: budget metrics are STRICT by default (cumulative chunk words must not exceed the budget; first chunk is not exempt). `bm*x` allows the chunk that crosses the budget. Relevance is article-scoped; retrieval searches the pooled chunks of the run's split.
