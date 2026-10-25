import re

p = "paper/paper.tex"
s = open(p, encoding="utf-8").read()

ABSTRACT = r"""Retrieval systems are shaped by how documents are cut into chunks. Most of what we know about
that choice comes from English, and most of it measures size in model tokens. This paper moves the
question to Urdu. We build a parallel Urdu--English corpus from UQA and SQuAD2.0: 150 articles and
7{,}362 test questions. Four multilingual encoders, BM25, and BM25+dense RRF run over five chunkers,
word sizes from 50 to 800, token budgets from 64 to 256, and three overlap levels. Budget-matched
recall at 500 words is the metric; intervals come from an article-level bootstrap. Three results
stand out. Word-defined optima agree: every encoder picks 50 words, the smallest size tested.
Token-defined optima do not (64/128/256), because 128 tokens is about 96 words for XLM-R and 42 for
Qwen3 (fertility 1.34 vs.\ 3.07). Urdu-aware sentence splitting beats English-toolkit defaults by
5.5--6.9 points at 50 words; at 200 words the sign flips. A 25\% overlap raises recall by 2.6--7.2
points, against several English studies, while orthographic normalization changes nothing measurable
on this machine-translated corpus. English is easier on the parallel control but lands on the same
fixed-word optimum. Code, configurations, and per-question metrics are released."""

INTRO_RELATED = r"""\section{Introduction}

Urdu has more than 230 million speakers. Retrieval research has not kept up. Mr.\ TyDi covers
eleven languages \citep{zhang2021mr}, MIRACL eighteen \citep{zhang2022making}, mMARCO thirteen
\citep{bonifacio2021mmarco}; Urdu appears in none of them. The resources that do exist are recent:
UQA \citep{arif2024uqa}, Urdu MS MARCO \citep{butt2024enabling}, and a small cluster of papers on
alignment and Urdu question answering \citep{faheem-etal-2026-beyond,ahmad-etal-2026-drrag}. None of
them varies how documents are segmented.

Segmentation decides what a retriever can return. English work shows the stakes: size sweeps move
recall \citep{bhat2025rethinking,bennani2026a,wang2024searching}, semantic chunking rarely earns its
cost \citep{qu2024is}, and structure-aware segmenters help in some settings
\citep{yepes2024financial,singh2026parser}. Two questions remain open.

The first is the unit. Nearly every study sizes chunks in model tokens. Tokenizers do not treat
languages alike \citep{petrov2023language,ahia2023do,srivastava2026the}. For Urdu the gap is wide:
the XLM-R tokenizer behind bge-m3 and multilingual-e5 emits 1.34 tokens per word; Qwen3 emits 3.07.
A 256-token chunk is roughly 192 words for one family and 83 for the other. If the optimum is stable
in words but not in tokens, specify words. If it is stable in neither, tune per model.

The second question is whether English-toolkit defaults travel. Recursive splitters reach for blank
lines, line breaks, spaces, and the ASCII period. Urdu ends sentences with U+06D4 and questions with
U+061F; its orthography tolerates Arabic/Persian letter variants, zero-width joiners, and optional
diacritics. A language-aware chunker built for Khmer, the only one we found, did worse than plain
recursive splitting \citep{chhoun2026evaluation}. Whether that generalises is unknown.

We ask four questions. (Q1) How do size, overlap, and strategy interact with the retriever family
(BM25, dense, hybrid)? (Q2) Is the best size stable in words rather than model tokens across
tokenizers? (Q3) Do Urdu-aware boundaries and normalization beat English-toolkit defaults? (Q4) Does
the parallel English control select the same optimum? The evaluation is retrieval-only: no
generator, no LLM judge. Our contributions are:
\begin{itemize}
\item the first systematic chunking evaluation for Urdu retrieval, to our knowledge: 360 runs over a
  150-article pool and 7{,}362 test questions, five chunkers, four encoders, two sparse retrievers,
  and RRF;
\item a direct word-versus-token comparison across tokenizers with measured fertility: token budgets
  are not comparable across models, word budgets are;
\item evidence that Urdu-aware sentence splitting helps at the small sizes dense retrievers prefer
  and hurts at large sizes; 25\% overlap helps on this corpus;
\item a parallel English control on the same articles and questions, plus a null result for
  orthographic normalization on machine-translated Urdu.
\end{itemize}

\section{Related Work}

\paragraph{Chunking for retrieval.}
Fixed token windows are still the default. \citet{bhat2025rethinking} sweep 64--1024 tokens over six
English QA datasets; the optimum depends on answer type and encoder. \citet{bennani2026a} vary
50--500 tokens and 0/20\% overlap with SPLADE and a generator, and find overlap unhelpful.
\citet{wang2024searching} compare 128--2048 tokens inside a larger RAG pipeline. Semantic and
structure-aware chunkers have been compared at fixed sizes
\citep{qu2024is,zhou2026beyond,caspari2026when,merola2025reconstructing,gnther2024late}, and
granularity itself has been studied as propositions versus passages \citep{chen2023dense}. The
closest template is \citet{singh2026parser}: a factorial study of parsers, chunkers, and embedders
on English regulatory documents, analysed with mixed-effects models. None of this work is Urdu.
None compares word- and token-defined sizes across tokenizers.

\paragraph{Low-resource and multilingual retrieval.}
LaBSE \citep{feng2020languageagnostic}, multilingual E5 \citep{wang2024multilingual}, BGE-M3
\citep{chen2024m3embedding}, and Qwen3 Embedding \citep{zhang2025qwen3} all claim Urdu support;
MMTEB \citep{enevoldsen2025mmteb} covers a lot of ground. Averages mislead. On Amharic, the best
zero-shot multilingual retriever trails a monolingual model by 23\% relative MRR, and BM25 beats
four of five zero-shot dense retrievers \citep{alemneh2026the}. Hubness and cross-lingual asymmetry
distort rankings in other languages \citep{sakhawat2026hubness,goworek2025what}. Urdu work has gone
into alignment \citep{faheem-etal-2026-beyond}, dual-representation retrieval
\citep{ahmad-etal-2026-drrag}, and QA data \citep{arif2024uqa,butt2024enabling}. Segmentation has
not been touched.

\paragraph{Tokenization and Urdu text.}
Tokenizers charge different prices for the same content \citep{petrov2023language,ahia2023do}, and
fertility predicts downstream accuracy in some settings \citep{lundin2025the}. Indic scripts pay a
large tax, partly structural \citep{srivastava2026the,regmi2026vowel}; single-token retention
exposes how the vocabulary is allocated \citep{nayeem2025beyond}. Urdu's literature is thinner.
CURE built an IR test collection with stemming and query expansion \citep{iqbal2020cure}; early
indexing work described normalization rules but did not test them \citep{qureshi2021an}. UQA
provides machine-translated SQuAD2.0 with preserved spans \citep{arif2024uqa}; Urdu MS MARCO
provides translated passages \citep{butt2024enabling}. We use UQA because it gives character-level
answer spans in parallel with English SQuAD2.0.

"""

# abstract
a = s.index("Chunking is a first-order design choice")
b = s.index("per-question metrics.", a) + len("per-question metrics.")
s = s[:a] + ABSTRACT + s[b:]

# intro + related
a = s.index(r"\section{Introduction}")
b = s.index(r"\section{Data}")
s = s[:a] + INTRO_RELATED + s[b:]

open(p, "w", encoding="utf-8").write(s)
print("ok", len(s))
