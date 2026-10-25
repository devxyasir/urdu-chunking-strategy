# Publication readiness and venue assessment

## Honest verdict

**Suitable for public release as a preliminary preprint, with limitations disclosed. Not yet ready
for submission to a peer-reviewed venue as a chunking-optimum paper.**

Reasons:

1. The 360-run package is internally consistent and the main tables reproduce. The paper no longer
   claims an interior optimum; it reports a boundary result. That is enough for a careful preprint.
2. The lower-size check is complete only for bge-m3 (50 remains best, curve monotonic). e5, Qwen3,
   and LaBSE still stop at 50, and the raw E1 per-question arrays are not in the workspace. A
   reviewer can still ask whether 50 is the grid floor for those three encoders.
3. The public artifact now exists at https://github.com/devxyasir/urdu-chunking-strategy. That reason
   is resolved; licence and DOI are not.
4. The corpus is machine-translated, single-domain, and has no native or human evaluation. These are
   disclosed, not solved.
5. UQA has no explicit licence. The public artifact must exclude UQA text; that is a release task, not
   a paper task.

After completing E1 for all four encoders and a public code/results artifact, the paper would be a
credible workshop or benchmark-track submission. Without them, it is a useful preprint and a complete
negative/conditional result, not a submission-ready optimum study.

## What changed in this revision

- Appendix experiment arithmetic corrected from 336 to **360**, with the missing BM25 normalization
  and bge/Qwen 400/800 runs itemized.
- "First systematic" removed; contribution narrowed to a systematic budget-matched evaluation.
- "Optimum" language replaced with "best tested size" in the abstract, results, discussion, and
  conclusion; the boundary is stated in Limitations.
- Truncation claim corrected: 3 of 9,800 LaBSE sentence-aware 100-word chunks (0.03%) exceed the
  256-token window; all other dense runs are clean.
- Token result rephrased as differing argmaxes with tied adjacent budgets, not a sharp disagreement.
- Overlap and strategy claims now name the four systems where the sweep was run; LaBSE is excluded
  from those sweeps and the text says so.
- "Every system" and "every model" claims narrowed; the English control is described as a confounded
  language/translation comparison.
- Limitations added: native queries, no human judgments, selection/multiplicity, private artifact.
- AI-use disclosure broadened; bootstrap seed 0, 10,000 resamples, model revisions, software and GPU
  recorded in the appendix.
- Public-release claim changed from "released" to "prepared for release; private during review."
- Guide updated to match the paper's wording.

## Remaining checklist

### Must do before a public preprint

- [x] Build a staging artifact with no UQA text (`public_artifact/`, 80.6 MB).
- [x] Remove corpus/question text from the staging artifact; verify no banned files.
- [x] Add `question_ids.csv` and a minimal `questions.csv` (IDs/splits only).
- [x] Add `requirements-full.txt` and record the archived CUDA/PyTorch versions.
- [ ] Choose a code licence (MIT/Apache-2.0) and a paper licence (CC BY 4.0); do not license UQA text.
- [x] Create the public GitHub repository and push `public_artifact/`
      (https://github.com/devxyasir/urdu-chunking-strategy).
- [x] Replace the private URL in `README.md`/`CITATION.cff` with the public URL.
- [ ] Archive the artifact with a DOI (Zenodo/OSF) and put the DOI in the paper.
- [ ] Recompile and proofread the PDF after the artifact URL is added.
- [ ] Check the target venue's AI-disclosure and preprint rules; the current statement is factual but
      venue-specific forms may be required.

### Must do before a peer-reviewed submission

- [~] E1: bge-m3 10/25/40/50 done (50 remains best); e5/Qwen3/LaBSE still stop at 50. The paper now
      reports the partial check and states the boundary honestly.
- [x] Self-containedness audit done with three raters: 77/100 majority self-contained, Fleiss
      kappa 0.734; one unclear row and one no-majority row disclosed.
- [ ] Decide whether to add LaBSE overlap/recursive runs or keep the four-system scope (currently
      stated).
- [ ] Add a native-authored set or keep every claim MT-scoped (currently stated).
- [ ] Add human context judgments or keep the paper retrieval-only (currently stated).
- [ ] Check reference formatting and remove the 12 uncited bib entries if the venue requires a clean
      bibliography.
- [ ] Re-run `analyze_full.py` and `make_figures.py` from the public artifact and compare hashes.

### Optional, high value

- [ ] E1 plus a 30/35/45 follow-up only if the pre-registered rule requires it.
- [ ] Translation-variant sensitivity (E5).
- [ ] A small human audit of 100 contexts (E4).
- [ ] A native 150--300 question set (E3), which would be a separate paper.

## Venue assessment

The work sits between an empirical IR paper and a benchmark/resource paper. It is not a new model or
a new dataset, and it has no generation evaluation.

| Option | Fit | What to check |
| --- | --- | --- |
| arXiv `cs.CL` preprint | Good now, with limitations | Endorsement/moderation; no fee; no peer review; do not call it published |
| ACL/NAACL/EMNLP main or Findings | Possible after all-encoder E1 and artifact | Current CFP; anonymity/preprint policy; no submission fee for main conference; open-access proceedings |
| LREC-COLING or a language-resources track | Good fit for the corpus/protocol, not for an optimum claim | Current CFP; fees; whether a private artifact is acceptable |
| Workshop on low-resource/multilingual NLP | Good fit for a modest, careful result | Current workshop scope; workshop fees; preprint policy |
| TACL or an IR journal | Premature | Requires a stronger contribution than a boundary result; check APC/open-access options |
| Technical report + open-source release | Honest alternative | No peer review; useful if the artifact is the main contribution |

Do not claim a DOI or arXiv posting is peer-reviewed publication. Do not submit the same manuscript
to two venues. Check the current ACL Rolling Review policy on preprints and anonymity before posting;
policies change, and this note is not legal or venue advice.

## Fees and access

- arXiv: no author fee; moderation/endorsement requirements apply.
- ACL-family main conferences: no submission fee; proceedings are open access. Workshops and
  tutorials may charge registration; check the specific event.
- LREC-COLING: check the current CFP for registration/fee structure; proceedings are typically open
  access.
- Journals: APCs vary widely; do not assume a fee until the venue's current page says so.
- Zenodo/OSF: free for ordinary deposits; DOI minting is free.

## What not to claim in the release announcement

- "First" or "state of the art."
- "Optimal chunk size for Urdu."
- "Overlap improves RAG" (no generator).
- "Normalization does not matter for Urdu" (null on MT UQA only).
- "Code released" until the public artifact resolves.
- "Peer-reviewed" for a preprint or DOI.
