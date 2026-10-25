import numpy as np


def is_relevant(chunk_span, gold_span):
    return chunk_span[0] <= gold_span[0] and chunk_span[1] >= gold_span[1]


def is_relevant_doc(chunk_doc, chunk_span, qdoc, gold_span):
    return chunk_doc == qdoc and is_relevant(chunk_span, gold_span)


def relevance_flags(ranked_ids, chunk_spans, gold_span, chunk_docs=None, qdoc=None):
    if chunk_docs is None:
        return np.array([is_relevant(chunk_spans[i], gold_span) for i in ranked_ids], dtype=bool)
    return np.array([is_relevant_doc(chunk_docs[i], chunk_spans[i], qdoc, gold_span) for i in ranked_ids], dtype=bool)


def recall_at_k(flags, k):
    return 1.0 if flags[:k].any() else 0.0


def mrr_at_10(flags):
    top = flags[:10]
    idx = np.flatnonzero(top)
    return 1.0 / (idx[0] + 1) if len(idx) else 0.0


def ndcg_at_10(flags):
    top = flags[:10]
    if not top.any():
        return 0.0
    gains = top.astype(float)
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(gains)))
    dcg = float((gains * discounts).sum())
    ideal = float(discounts[: int(top.sum())].sum())
    return dcg / ideal if ideal else 0.0


def bm_recall(ranked_ids, chunk_spans, chunk_words, gold_span, budget=500, chunk_docs=None, qdoc=None):
    used = 0
    hit = False
    for cid in ranked_ids:
        if used >= budget:
            break
        used += chunk_words[cid]
        if chunk_docs is None:
            rel = is_relevant(chunk_spans[cid], gold_span)
        else:
            rel = is_relevant_doc(chunk_docs[cid], chunk_spans[cid], qdoc, gold_span)
        if rel:
            hit = True
    return hit, used
