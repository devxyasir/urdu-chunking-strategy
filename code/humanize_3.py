p = "paper/paper.tex"
s = open(p, encoding="utf-8").read()

R51 = r"""\subsection{Size and retriever}
\label{sec:size}

Dense and sparse retrievers disagree about size (Figure~\ref{fig:size}, Table~\ref{tab:words}).
Every dense encoder does best at 50 words and then declines, with no second peak. BM25 climbs to
200--400 words and drops at 800. At its optimum bge-m3 reaches 0.773 (95\% CI 0.758--0.789), 6.5
points above BM25 at 0.708 (0.688--0.729). A sparse retriever wants a chunk four to eight times
longer than a dense one.

The fall is steep. bge-m3 loses 1.6 points from 50 to 100 and 7.7 from 100 to 200; e5 loses 2.4 and
8.9; Qwen3 loses 4.7 and 8.0. Every encoder's 50--100 drop is significant (bge +1.65 [0.53, 2.80],
e5 +2.40 [1.06, 3.71], Qwen3 +4.71 [3.35, 6.11], LaBSE +5.38 [3.89, 6.86]).

"""

R52 = r"""\subsection{Words versus model tokens}
\label{sec:unit}

All four encoders land on 50 words. Their token budgets do not agree: 128 for bge-m3, 64 for e5, 256
for Qwen3 (Table~\ref{tab:tokens}). The disagreement is softer than it looks. Neighbours are
statistically tied: bge 64 vs.\ 128 is $-0.08$ points [$-1.60$, 1.43]; e5 is $+0.35$ [$-1.08$,
1.75]; Qwen3 128 vs.\ 256 is $-0.09$ [$-1.54$, 1.33]. These are plateaus, not sharp breaks.

What a token means is the durable difference. At 128 tokens XLM-R sees about 96 words; Qwen3 sees
about 42 (Figure~\ref{fig:tokens}). At 256 tokens the gap is roughly 192 words against 83. Word
sizes remove that ambiguity. Token sizes quietly change the task from one model to the next.

"""

R53 = r"""\subsection{Boundaries and normalization}
\label{sec:norm}

At 50 words, sentence-aware splitting beats the English recursive default in every system
(Figure~\ref{fig:split}): bge-m3 +6.02 [4.98, 7.06], e5 +6.21 [5.18, 7.21], Qwen3 +6.90 [5.59,
8.15], BM25 +5.53 [4.31, 6.70]. Against fixed-size chunks the gain is larger: +11.4, +11.0, +12.7,
and +10.1 points. At 100 words it shrinks and turns mixed (bge +1.41 [0.30, 2.53]; e5 +0.86
[$-$0.32, 1.99]; Qwen3 $-$2.04 [$-3.34$, $-$0.74]). At 200 it reverses: bge $-$8.99, e5 $-$10.38,
Qwen3 $-$11.75, BM25 $-$3.09.

The answer-split rate shows part of the mechanism. Fixed 50-word chunks cut 4.10\% of answers;
sentence-aware 50-word chunks cut 0.05\%. At 200 words the rates are 1.06\% and 0.00\%. The reversal
is therefore not about splitting. The English recursive rule, applied to Urdu, produces smaller
units and a different retrieval granularity at large targets. This is an interaction, not a free win.

Normalization barely moves. N1 and N2 differ from N0 by at most 0.09 points. Every article-level CI
covers zero except a 0.085-point N2 gain for BM25 at 50 words ($p=0.038$). The Unicode audit
explains why: no ZWNJ or ZWJ, one tatweel, nine Arabic \texttt{yeh} characters, six Urdu digits. We
report the null instead of dressing it up.

"""

R54 = r"""\subsection{Overlap}

Against the English literature, 25\% overlap helps at every size and model
(Figure~\ref{fig:overlap}). bge-m3 gains 6.91 [5.72, 8.05] at 50 words, 5.24 [4.29, 6.18] at 100,
and 4.22 [3.10, 5.38] at 200. e5 gains 7.16, 5.13, and 3.77; Qwen3 6.13, 5.75, and 4.97. BM25
follows with 6.97, 5.51, and 2.60. Overlap keeps spans intact and gives 50-word chunks enough
surrounding text to rank. Both effects are visible in the per-question data. The result is
corpus-specific; we do not claim it travels.

"""

R55 = r"""\subsection{English control}

English is easier on the parallel questions, but the fixed-word optimum does not move: 50 words in
both languages (Table~\ref{tab:english}). The gap is 7.4 points at 50, 7.2 at 100, and 7.8 at 200.
For recursive-English chunks it is 10.8 at 50, 6.3 at 100, and gone at 200 ($-$0.9 [$-2.29$,
0.46]). Nothing here says the languages need different sizes. It does say the Urdu side is harder,
which fits translation artifacts and weaker Urdu coverage in the encoders.

"""

R56 = r"""\subsection{Best systems}

Sentence-aware 50-word chunks with bge-m3 score 0.888 [0.876, 0.899]. e5 is 0.886 [0.873, 0.897];
the two are tied. RRF reaches 0.871 [0.857, 0.884] with bge and 0.865 with e5, below the best single
model. It does lift the weak ones: Qwen3 from 0.730 to 0.814, LaBSE from 0.703 to 0.810, BM25 from
0.749 to 0.871. BM25-char trails BM25-whitespace by 2--4 points at every size. Table~\ref{tab:best}
has the full ranking.

"""

DISC = r"""\section{Discussion}

\paragraph{Words are the portable unit; tokens are not.}
Specify chunk size in words when encoders use different tokenizers. A word is a language unit. A
token is a model unit whose size depends on the vocabulary. The measured fertility gap here is
2.3$\times$ (1.34 vs.\ 3.07). That is enough to turn one nominal configuration into two different
retrieval tasks. The token optima are plateaus, so the problem is not a sharp 64-versus-256
disagreement. It is that ``128 tokens'' does not name a fixed amount of text.

\paragraph{The word optimum sits at the grid boundary.}
Every dense curve falls from 50 words. The shared optimum is the smallest size tested, not an
interior one. The data show that the best \emph{tested} size is the same across encoders. They do
not show that 50 is optimal in any absolute sense. A finer grid (10/25/50) is the next step, and we
do not claim it here. The boundary also explains why Urdu-aware splitting gains most at 50 words:
sentence-aware chunks avoid the 4.1\% answer-split rate of fixed 50-word chunks.

\paragraph{Why overlap helps here.}
English chunking work often finds overlap neutral or harmful
\citep{bennani2026a,amiri2025chunk}. This corpus differs in two ways. Documents are concatenations
of short translated paragraphs, and answers are short spans. Overlap keeps a span intact when a
boundary would cut it, and it gives 50-word chunks context they otherwise lack. Four systems show
the same direction, so it is not one model's quirk. It remains a corpus-specific result.

\paragraph{Language-aware splitting is conditional.}
Urdu-aware splitting is not a free win. At 50 words it beats the English default by six points. At
200 words it loses by 9--12. A paper that reported one size could reach the opposite conclusion.
The Khmer result fits this pattern \citep{chhoun2026evaluation}: language-aware chunkers interact
with size and with the recursive merge policy. One size is not a test.

\paragraph{Normalization is a property of the corpus.}
The null is informative. UQA's translation pipeline already produces normalized Urdu: no ZWNJ,
essentially no tatweel, ASCII digits, and nine Arabic-variant \texttt{yeh} characters. A study on
naturally occurring Urdu web text could still find an effect. Here there is almost nothing to
normalize. We report the Unicode audit so the null can be read rather than guessed at.

"""

LIM = r"""\section{Limitations}

\begin{itemize}
\item \bb{Machine-translated corpus.} UQA is MT, with 4.86 variants per paragraph. The modal rule
  keeps 20\% of rows and may select a systematically different translation. The English control is
  original SQuAD text, so the language comparison mixes language with translation artifacts.
\item \bb{Boundary optimum.} 50 words is the smallest size. No interior optimum is identified.
\item \bb{Single domain.} Wikipedia articles made of concatenated paragraphs. Not web text, legal
  text, or conversation.
\item \bb{Retrieval only.} No generator and no LLM judge. The conclusions concern evidence
  retrieval, not answer quality.
\item \bb{Fixed BM25.} $k_1$ and $b$ were fixed in advance, not tuned. A tuned BM25 is a different
  experiment.
\item \bb{Small normalization exclusions.} 37 of 9{,}077 questions (0.41\%) fail N1/N2 span
  mapping because UQA answers are whitespace-padded. They are excluded from those variants only.
\item \bb{Prefix sensitivity.} Qwen3's documented instruction prefix lowered BM500 relative to no
  prefix (0.400 vs.\ 0.460). Its absolute numbers may understate the model.
\end{itemize}

"""

ETH = r"""\section{Ethics and Reproducibility}

SQuAD2.0 is CC BY-SA 4.0. UQA has no explicit licence, so we release code, configurations, and item
IDs rather than UQA text. The corpus is encyclopedic and contains no personal or sensitive data. No
LLM generated evaluation data, labels, or rankings. Every question and span comes from the released
corpora, and every metric is deterministic. We report 360 runs, fixed seeds, model revisions,
per-question metrics, and article-level CIs. AI assistance was limited to coding and editing, and we
disclose it under the venue's policy.

"""

CONC = r"""\section{Conclusion}

For Urdu retrieval, define chunk size in words, not model tokens. The same token budget is 96 words
for XLM-R and 42 for Qwen3, while the best tested word size is 50 for every encoder. Urdu-aware
sentence splitting helps at the sizes dense retrievers prefer (5.5--6.9 points) and reverses at 200
words. A 25\% overlap helps on this corpus. Normalization is a null on machine-translated Urdu. We
release the pipeline and per-question results so the size-by-unit interaction can be tested on other
languages and on naturally occurring text.

"""

APP = r"""\appendix
\section{Additional results}

\paragraph{Strict versus crossing budget.}
The main text uses the crossing definition. Under the strict definition the BM25 argmax moves from
200 to 100 words (0.686 at 100, 0.663 at 200). Every dense argmax stays at 50. No dense conclusion
changes.

\paragraph{Normalization.}
N0/N1/N2 means for BM25-whitespace at 50/100/200 words differ by at most 0.09 points. The largest
difference is N2 at 50 words (+0.085, $p=0.038$). N1/N2 exclude 37 whitespace-padded answers.

\paragraph{Prefix checks.}
On 100 dev questions at 200 words, the e5 instruction prefix improves BM500 from 0.550 to 0.640 and
R@1 from 0.43 to 0.49. The Qwen3 prefix changes BM500 from 0.460 to 0.400 and R@1 from 0.31 to 0.32.
Both models use their documented format in every reported run.

\paragraph{Grid.}
BM25: 2 sparse variants $\times$ 5 strategies $\times$ 5 sizes $\times$ 3 overlaps = 150 runs. Dense:
21 configurations per model for bge, e5 and Qwen3; 6 for LaBSE; 12 extra 400/800 configurations for
bge and Qwen3; 21 English runs; 84 RRF runs. Total 360.

"""

# Results blocks, bottom-up
blocks = [
    (r"\subsection{Size and retriever}", r"\begin{figure*}[t]", R51),
    (r"\subsection{Words versus model tokens}", r"\begin{figure*}[t]", R52),
    (r"\subsection{Boundaries and normalization}", r"\begin{figure}[t]", R53),
    (r"\subsection{Overlap}", r"\begin{figure*}[t]", R54),
    (r"\subsection{English control}", r"\begin{table}[t]", R55),
    (r"\subsection{Best systems}", r"\begin{table}[t]", R56),
]
spans = []
for start, end, new in blocks:
    a = s.index(start)
    b = s.index(end, a)
    spans.append((a, b, new))
for a, b, new in sorted(spans, reverse=True):
    s = s[:a] + new + s[b:]

# Whole sections
for start, end, new in [
    (r"\section{Discussion}", r"\section{Limitations}", DISC),
    (r"\section{Limitations}", r"\section{Ethics and Reproducibility}", LIM),
    (r"\section{Ethics and Reproducibility}", r"\section{Conclusion}", ETH),
    (r"\section{Conclusion}", r"\bibliography{refs}", CONC),
    (r"\appendix", r"\end{document}", APP),
]:
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]

open(p, "w", encoding="utf-8").write(s)
print("ok", len(s))
