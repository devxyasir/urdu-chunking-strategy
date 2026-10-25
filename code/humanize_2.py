p = "paper/paper.tex"
s = open(p, encoding="utf-8").read()

PARALLEL = r"""\paragraph{Parallel corpus.}
We start from UQA \citep{arif2024uqa}, a machine translation of SQuAD2.0 into Urdu with preserved
answer spans, and from the original SQuAD2.0 articles. The released UQA has 124{,}745 training rows
mapped to 19{,}021 English paragraphs. One property mattered more than the rest, and we measured it
rather than assuming it: each English paragraph was translated again and again. The average
paragraph carries 4.86 distinct Urdu contexts; the maximum is 18. Pairs we inspected sit at
0.96--0.98 string similarity.

Documents are built with a \emph{modal-variant} rule. For every English paragraph we keep the most
frequent Urdu context (ties go to the longer string, then the lexicographically larger one). A
question survives only when its row uses that variant and the question is answerable. An article
document is the chosen paragraphs joined in order. Of 124{,}745 rows, 24{,}898 remain (20.0\%).
Every retained span was checked: 83{,}018/83{,}018 train and 11{,}169/11{,}169 validation rows pass
the character-offset equality test. The English control reuses the same articles and questions with
SQuAD paragraphs and offsets.

"""

POOL = r"""\paragraph{Pool and splits.}
We keep the 403 articles with at least 20 retained questions, 20 paragraphs, and 1{,}500 Urdu words.
They are sorted by SHA-1 of the title, and the first 150 are taken. Every fifth article is
development (30 articles); the other 120 form the test set. A question-level split would leak
chunks across the boundary, so the split is by article. Table~\ref{tab:data} gives the totals.

"""

BOUND = r"""\paragraph{Urdu boundaries and tokenization.}
Figure~\ref{fig:example} shows the linguistic contrast that drives the study: a two-sentence Urdu
passage whose word count is the same for every tokenizer but whose token count differs by
3$\times$. The study corpus holds 34{,}737 U+06D4 boundaries and only 51 U+061F question marks,
against 2{,}310 ASCII periods. Zero-width joiners are absent, tatweel appears once, and there are
nine Arabic \texttt{yeh} characters and six Urdu digits. Normalization has almost nothing to change
(Section~\ref{sec:norm}).

"""

METHOD = r"""\section{Method}

\paragraph{Chunkers.}
All five chunkers return character offsets in the normalized document.
\emph{Fixed words} slides a window of $s$ whitespace words with step
$\max(1,\mathrm{round}(s(1-o)))$.
\emph{Fixed tokens} packs whole words while the model tokenizer's content-token count stays within
$b$; it steps by the same overlap fraction in words and never cuts inside a word.
\emph{Sentence-aware} splits at U+06D4, U+061F, and the exclamation mark when whitespace or a
paragraph break follows, then groups whole sentences up to a word target. Only a sentence longer
than twice the target is split by words.
\emph{Recursive-English} follows the English-toolkit separators (paragraph, newline, ASCII period
followed by whitespace and an uppercase letter or end, question mark, exclamation mark, space) and
merges units up to the target.
\emph{Recursive-Urdu} runs the same merge with Urdu terminators and paragraph breaks.
At zero overlap every chunker covers every non-whitespace character exactly once. We test that on
ten documents for every configuration: 1{,}170 checks, zero uncovered characters.

\paragraph{Retrievers.}
BM25 tokenizes on whitespace, strips Urdu and ASCII punctuation, and uses $k_1=1.2$ and $b=0.75$.
Both values were fixed in advance. BM25-char works on 3--5 character n-grams generated inside
words. Dense retrieval is cosine similarity over L2-normalized vectors from bge-m3
\citep{chen2024m3embedding}, multilingual-e5-large-instruct \citep{wang2024multilingual},
Qwen3-Embedding-0.6B \citep{zhang2025qwen3}, and LaBSE \citep{feng2020languageagnostic}. Queries use
the documented instruction formats; the same format is applied to every run. RRF fuses
BM25-whitespace and the dense ranking of the same chunk configuration with $k=60$.

\paragraph{Metrics.}
A chunk counts as relevant when it contains the full gold answer span. We report Recall@1/3/5/10,
MRR@10, nDCG@10, and the primary budget-matched recall. That last one walks the ranking, adds chunk
word counts until a budget $B$ is reached, and scores 1 if at least one retrieved chunk contains the
full answer. A span split across chunks is a failure, and we report the answer-split rate next to
recall. The primary budget is $B=500$ words; 200, 1000, and 2000 are secondary. The crossing
definition, which includes the chunk that reaches the budget, is the pre-registered one. The strict
definition appears in the appendix; it changes only the BM25 argmax.

\paragraph{Statistics.}
The analysis unit is the article, not the question. Every question in an article shares the same
chunk set, so averaging over questions would pretend to a precision the design does not have. For
each configuration we compute a per-article mean, then average over the 120 test articles.
Confidence intervals come from 10{,}000 bootstrap resamples over articles. Paired contrasts reuse
the same resample for both configurations. The arithmetic is blunt: 61 questions share one chunk
set, and treating them as independent reports power 1.00 for a 5-point effect that the article-level
analysis detects with about 75\% power.

\paragraph{Grid.}
BM25 covers five strategies, five word sizes (50/100/200/400/800), and three overlaps (0/10/25\%).
Dense models cover 21 configurations: fixed words at 50/100/200 with 0/10/25\% overlap (9), fixed
tokens at 64/128/256 (3), and sentence-aware, recursive-English, and recursive-Urdu at 50/100/200
with zero overlap (9). bge-m3 and Qwen3 also run 400/800 words with 0/25\% overlap. LaBSE is
secondary and runs at 50/100 words. e5 stops at 200 words: its 512-token window truncates 74\% of
400-word chunks and 100\% of 800-word chunks. No dense run reports a truncated chunk. The full study
is 360 runs.

"""

# Data paragraph blocks (apply bottom-up so earlier anchors stay valid)
a_par, b_par = s.index(r"\paragraph{Parallel corpus.}"), s.index(r"\paragraph{Pool and splits.}")
a_pool, b_pool = s.index(r"\paragraph{Pool and splits.}"), s.index(r"\begin{table}[t]")
a_bound, b_bound = s.index(r"\paragraph{Urdu boundaries and tokenization.}"), s.index(r"\begin{figure*}[t]")

s = s[:a_bound] + BOUND + s[b_bound:]
s = s[:a_pool] + POOL + s[b_pool:]
s = s[:a_par] + PARALLEL + s[b_par:]

# Method whole section
a = s.index(r"\section{Method}")
b = s.index(r"\section{Results}")
s = s[:a] + METHOD + s[b:]

open(p, "w", encoding="utf-8").write(s)
print("ok", len(s))
