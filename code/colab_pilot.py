import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

SQUAD_URL = "https://raw.githubusercontent.com/sameearif/UQA/main/SQuAD/train-v2.0.json"
UQA_URL = "https://huggingface.co/datasets/uqa/UQA/resolve/main/data/train-00000-of-00001-bac007e8ca719235.parquet"


def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        urllib.request.urlretrieve(url, path)
    return path


print("downloading data")
fetch(SQUAD_URL, "squad_train.json")
fetch(UQA_URL, "uqa_train.parquet")

import pyarrow.parquet as pq

squad = json.load(open("squad_train.json", encoding="utf-8"))
rows = pq.read_table("uqa_train.parquet").to_pylist()
print("rows", len(rows))

id2key, eng, order = {}, {}, {}
for art in squad["data"]:
    t = art["title"]
    order[t] = []
    for pi, p in enumerate(art["paragraphs"]):
        key = (t, pi)
        order[t].append(key)
        eng[key] = p["context"]
        for qa in p["qas"]:
            id2key[qa["id"]] = key

groups = defaultdict(list)
for r in rows:
    k = id2key.get(r["id"])
    if k is not None:
        groups[k].append(r)

chosen = {}
for k, rs in groups.items():
    cnt = Counter(r["context"] for r in rs)
    chosen[k] = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]

ret = []
for r in rows:
    k = id2key.get(r["id"])
    if k is not None and not r["is_impossible"] and r["context"] == chosen[k]:
        ret.append(r)

q_by_title = Counter(id2key[r["id"]][0] for r in ret)
words_by_title = defaultdict(int)
paras_by_title = defaultdict(int)
for k, ctx in chosen.items():
    words_by_title[k[0]] += len(ctx.split())
    paras_by_title[k[0]] += 1

eligible = [t for t in order if q_by_title[t] >= 20 and paras_by_title[t] >= 20 and words_by_title[t] >= 1500]
eligible.sort(key=lambda t: hashlib.sha1(t.encode()).hexdigest())
pool = eligible[:150]
split = {t: ("dev" if i % 5 == 0 else "test") for i, t in enumerate(pool)}
dev_titles = [t for t in pool if split[t] == "dev"]
print("dev articles", len(dev_titles))

qrows = defaultdict(list)
for r in ret:
    qrows[id2key[r["id"]]].append(r)

docs = {}
for t in dev_titles:
    paras = [chosen[k] for k in order[t] if k in chosen]
    text = "\n\n".join(paras)
    offs, pos = [], 0
    for p in paras:
        offs.append(pos)
        pos += len(p) + 2
    qs = []
    for pi, k in enumerate([k for k in order[t] if k in chosen]):
        for r in qrows.get(k, []):
            st = int(r["answer_start"])
            ans = r["answer"]
            assert chosen[k][st:st + len(ans)] == ans
            ds = offs[pi] + st
            qs.append({"q": r["question"], "a": ans, "s": ds, "e": ds + len(ans)})
    docs[t] = {"text": text, "qs": qs}

rng = np.random.default_rng(0)
allq = [(t, i) for t in dev_titles for i in range(len(docs[t]["qs"]))]
keep = sorted(rng.choice(len(allq), 500, replace=False))
QUESTIONS = [(docs[t]["qs"][i], t) for t, i in (allq[k] for k in keep)]
print("questions", len(QUESTIONS))

WORD_RE = re.compile(r"\S+")


def words(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def fixed_words(text, size, ov):
    w = words(text)
    step = max(1, int(round(size * (1 - ov))))
    out, i = [], 0
    while i < len(w):
        j = min(i + size, len(w))
        out.append((w[i][0], w[j - 1][1]))
        if j >= len(w):
            break
        i += step
    return out


def paras(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]


def urdu_sents(text, s, e):
    out, start, i = [], s, s
    while i < e:
        if text[i] in "\u06d4\u061f!" and (i + 1 == e or text[i + 1].isspace()):
            out.append((start, i + 1))
            start = i + 1
            while start < e and text[start].isspace():
                start += 1
            i = start
        else:
            i += 1
    if start < e and text[start:e].strip():
        out.append((start, e))
    return out


def english_sents(text, s, e):
    out, start, i = [], s, s
    while i < e:
        ch = text[i]
        ok = ch in "?!" or (ch == "." and (i + 2 >= e or (text[i + 1].isspace() and i + 2 < e and text[i + 2].isupper())))
        if ok:
            out.append((start, i + 1))
            start = i + 1
            while start < e and text[start].isspace():
                start += 1
            i = start
        else:
            i += 1
    if start < e and text[start:e].strip():
        out.append((start, e))
    return out


def split_big(text, span, size):
    s, e = span
    w = [(m.start(), m.end()) for m in WORD_RE.finditer(text, s, e)]
    if len(w) <= size:
        return [span]
    return [(w[i][0], w[min(i + size, len(w)) - 1][1]) for i in range(0, len(w), size)]


def units(text, mode, size):
    out = []
    for ps, pe in paras(text):
        ss = urdu_sents(text, ps, pe) if mode == "urdu" else english_sents(text, ps, pe)
        for sp in ss:
            out.extend(split_big(text, sp, size))
    return out


def pack(text, us, size, ov):
    cnt = [len(WORD_RE.findall(text[s:e])) for s, e in us]
    out, i, n = [], 0, len(us)
    while i < n:
        j, used = i, 0
        while j < n and (used == 0 or used + cnt[j] <= size):
            used += cnt[j]
            j += 1
        out.append((us[i][0], us[j - 1][1]))
        if j >= n:
            break
        target, back, acc = size * ov, 0, 0
        while j - back - 1 > i and acc < target:
            back += 1
            acc += cnt[j - back]
        i = j - back
    return out


def sentence_aware(text, size, ov):
    return pack(text, units(text, "urdu", 10 ** 9), size, ov)


def recursive_english(text, size, ov):
    return pack(text, units(text, "english", size), size, ov)


def recursive_urdu(text, size, ov):
    return pack(text, units(text, "urdu", size), size, ov)


CH = {"fixed_words": fixed_words, "sentence_aware": sentence_aware,
      "recursive_english": recursive_english, "recursive_urdu": recursive_urdu}


def tokenize(t):
    return [w.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\u06d4\u061f\u060c\u061b') for w in t.split() if w.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\u06d4\u061f\u060c\u061b')]


class BM25:
    def __init__(self, toks, k1=1.2, b=0.75):
        self.k1, self.b, self.n = k1, b, len(toks)
        self.dl = np.array([len(t) for t in toks], dtype=np.float64)
        self.avgdl = self.dl.mean() if self.n else 0
        post = {}
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                post.setdefault(term, ([], []))
                post[term][0].append(i)
                post[term][1].append(tf)
        self.post = {t: (np.array(d, np.int32), np.array(f, np.float64)) for t, (d, f) in post.items()}
        self.idf = {t: float(np.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5))) for t, p in self.post.items()}
        self.norm = self.k1 * (1 - self.b + self.b * self.dl / (self.avgdl or 1))

    def topk(self, q, k=100):
        s = np.zeros(self.n)
        for term in set(q):
            if term in self.post:
                d, f = self.post[term]
                s[d] += self.idf[term] * f * (self.k1 + 1) / (f + self.norm[d])
        idx = np.argsort(-s)[:k] if k < self.n else np.argsort(-s)
        return idx


def rel(cdoc, cspan, qdoc, gold):
    return cdoc == qdoc and cspan[0] <= gold[0] and cspan[1] >= gold[1]


def ev(ranked, spans, docs_, words_, qdoc, gold, budget=500):
    used, hit, flags = 0, False, []
    for cid in ranked:
        r = rel(docs_[cid], spans[cid], qdoc, gold)
        flags.append(r)
        if used < budget:
            used += words_[cid]
            hit = hit or r
    flags = np.array(flags)
    r1 = float(flags[:1].any())
    r10 = float(flags[:10].any())
    mrr = 1.0 / (np.flatnonzero(flags[:10])[0] + 1) if flags[:10].any() else 0.0
    return r1, r10, mrr, float(hit), float(not flags.any())


CONFIGS = [("fixed_words", 50, 0.0), ("fixed_words", 100, 0.0), ("fixed_words", 200, 0.0),
           ("recursive_urdu", 100, 0.0), ("recursive_english", 100, 0.0), ("sentence_aware", 100, 0.0)]

print("installing sentence-transformers")
try:
    import sentence_transformers
except Exception:
    os.system("pip -q install sentence-transformers")
from sentence_transformers import SentenceTransformer
print("loading model")
model = SentenceTransformer("BAAI/bge-m3", device="cuda")

results = []
t0 = time.time()
for strat, size, ov in CONFIGS:
    texts, spans, cdocs = [], [], []
    for ai, t in enumerate(dev_titles):
        for s, e in CH[strat](docs[t]["text"], size, ov):
            texts.append(docs[t]["text"][s:e])
            spans.append((s, e))
            cdocs.append(ai)
    index = BM25([tokenize(x) for x in texts])
    cwords = [len(x.split()) for x in texts]
    bm = []
    for q, t in QUESTIONS:
        ai = dev_titles.index(t)
        ids = index.topk(tokenize(q["q"]), 100)
        bm.append(ev(ids, spans, cdocs, cwords, ai, (q["s"], q["e"])))
    t0e = time.time()
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    qemb = model.encode([q["q"] for q, _ in QUESTIONS], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    scores = qemb @ emb.T
    dn = []
    for qi, (q, t) in enumerate(QUESTIONS):
        ids = np.argsort(-scores[qi])[:100]
        dn.append(ev(ids, spans, cdocs, cwords, dev_titles.index(t), (q["s"], q["e"])))
    row = {"strategy": strat, "size": size, "chunks": len(texts)}
    for name, arr in [("bm25", bm), ("dense", dn)]:
        a = np.array(arr)
        row[name] = {"r1": float(a[:, 0].mean()), "r10": float(a[:, 1].mean()),
                     "mrr": float(a[:, 2].mean()), "bm500": float(a[:, 3].mean()),
                     "split": float(a[:, 4].mean())}
    results.append(row)
    print(strat, size, row["bm25"], row["dense"], "emb_sec", round(time.time() - t0e, 1), flush=True)

out = {"model": "BAAI/bge-m3", "split": "dev", "n_questions": len(QUESTIONS), "n_articles": len(dev_titles),
       "results": results, "seconds": round(time.time() - t0, 1)}
json.dump(out, open("pilot_bge_m3.json", "w"), indent=2)
print(json.dumps(out["results"], indent=1))
from google.colab import files
files.download("pilot_bge_m3.json")
