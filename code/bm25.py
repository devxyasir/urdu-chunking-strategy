import re
from collections import Counter

import numpy as np

PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
URDU_PUNCT = set("\u06d4\u061f\u060c\u061b\u06d4\u061f\u00ab\u00bb\u201c\u201d\u2018\u2019")


def tokenize_ws(text):
    toks = []
    for w in text.split():
        w = w.strip("".join(PUNCT | URDU_PUNCT))
        if w:
            toks.append(w.lower())
    return toks


def tokenize_char(text, nmin=3, nmax=5):
    grams = []
    for w in text.split():
        w = w.lower()
        if len(w) < nmin:
            grams.append(w)
            continue
        for n in range(nmin, nmax + 1):
            for i in range(len(w) - n + 1):
                grams.append(w[i:i + n])
    return grams


class BM25:
    def __init__(self, doc_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(doc_tokens)
        self.dl = np.array([len(d) for d in doc_tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        postings = {}
        for i, toks in enumerate(doc_tokens):
            for term, tf in Counter(toks).items():
                postings.setdefault(term, ([], []))
                postings[term][0].append(i)
                postings[term][1].append(tf)
        self.post = {}
        for term, (docs, tfs) in postings.items():
            self.post[term] = (np.array(docs, dtype=np.int32), np.array(tfs, dtype=np.float64))
        self.norm = self.k1 * (1 - self.b + self.b * self.dl / (self.avgdl or 1.0))
        self.idf = {}
        for term, (docs, _) in self.post.items():
            df = len(docs)
            self.idf[term] = float(np.log(1 + (self.n - df + 0.5) / (df + 0.5)))

    def score(self, query_tokens):
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(query_tokens):
            if term not in self.post:
                continue
            docs, tfs = self.post[term]
            contrib = self.idf[term] * tfs * (self.k1 + 1) / (tfs + self.norm[docs])
            scores[docs] += contrib
        return scores

    def topk(self, query_tokens, k=100):
        s = self.score(query_tokens)
        if k >= len(s):
            idx = np.argsort(-s)
        else:
            idx = np.argpartition(-s, k)[:k]
            idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]
