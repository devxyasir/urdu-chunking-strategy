# Artifact map

| Paper item | File |
| --- | --- |
| 360 runs, counts | `data/external/urdu_chunking_full/run_index.jsonl` |
| Per-question metrics | `data/external/urdu_chunking_full/runs.jsonl.gz` |
| Tables 2--5 | `results/analysis/` after `analyze_full.py` |
| Figures | `figures/` after `make_figures.py` |
| Run count audit | `code/audit_verify.py` |
| Lower-size plan/code | `docs/audit/revised_experiments.md`, `code/run_small_sizes.py` |
| Corrections | `data/external/urdu_chunking_full/manifest.json` (`corrections`, `settings_original`) |

The UQA corpus and question/answer files are not included. `questions.csv` has only `qid,article_id,split`.
