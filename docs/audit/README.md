# Audit package index

This directory contains the review requested for the Urdu chunking manuscript. Start with the audit
and the verdict; the other files are the evidence and the follow-up plan.

| File | Deliverable | What it answers |
| --- | --- | --- |
| `research_audit.md` | A. Research audit | Prioritized critical, major, and minor issues, with concrete fixes |
| `verification_report.md` | B. Evidence and verification | Run counts, table reproduction, citation keys, what is verified vs not |
| `novelty_assessment.md` | C. Literature and novelty | Comparison table and a defensible contribution statement |
| `revised_experiments.md` | D. Revised experimental plan | The 10/25/40/50 dense grid, optional studies, decision rules |
| `reproducibility_plan.md` | E. Reproducibility plan | Public artifact structure, commands, licences, blockers |
| `publication_readiness.md` | G/H. Checklist and verdict | Remaining steps, venue fit, honest classification |
| `optional_studies.md` | Optional | Native-query and human-context protocols, if you widen the scope |
| `../../public_artifact/` | Public staging | 80.6 MB, no UQA text; built by `code/build_public_artifact.py` |
| `../paper/paper_preprint.pdf` | F. Revised paper | Named preprint, 10 pages, claims narrowed |
| `../paper/paper.pdf` | F. Revised paper (anonymous) | Same content with review formatting |
| `../guide/urdu_chunking_guide.pdf` | Companion | 41-page defense guide, updated wording |

## One-line verdict

Suitable for public release as a preliminary preprint with limitations disclosed; not yet ready for
a peer-reviewed submission as a chunking-optimum paper until the lower-size dense grid and a public
artifact exist.

## Verification in one line

360 unique runs (162 BM25, 114 dense, 84 RRF); all per-question arrays have 7,362 entries; Tables 2--5
reproduce from the archived runs to within 0.0001. The appendix's 336 was an arithmetic error, not a
run-inventory error; the corrected total is 360.

## What is still open

1. Dense 10/25/40/50 grid (E1) — not run; no GPU locally.
2. Public code/results artifact — pushed to
   https://github.com/devxyasir/urdu-chunking-strategy; licence and DOI still open.
3. Native Urdu set — does not exist.
4. Human context judgments — do not exist.
5. LaBSE overlap/recursive cells — not run; claims now name the four systems where the sweep ran.
