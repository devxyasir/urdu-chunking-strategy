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
WORK = "/content/urdu_chunking"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)


def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print("downloading", path, flush=True)
        urllib.request.urlretrieve(url, path)
    return path


fetch(SQUAD_URL, "squad_train.json")
fetch(UQA_URL, "uqa_train.parquet")

import pyarrow.parquet as pq

squad = json.load(open("squad_train.json", encoding="utf-8"))
rows = pq.read_table("uqa_train.parquet").to_pylist()
print("UQA rows:", len(rows), flush=True)

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

ret = [r for r in rows if id2key.get(r["id"]) is not None and not r["is_impossible"] and r["context"] == chosen[id2key[r["id"]]]]

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
print("pool articles:", len(pool), "dev:", len(dev_titles), flush=True)

qrows = defaultdict(list)
for r in ret:
    qrows[id2key[r["id"]]].append(r)

docs = {}
for t in dev_titles:
    keys = [k for k in order[t] if k in chosen]
    paras = [chosen[k] for k in keys]
    text = "\n\n".join(paras)
    offs, pos = [], 0
    for p in paras:
        offs.append(pos)
        pos += len(p) + 2
    qs = []
    for pi, k in enumerate(keys):
        for r in qrows.get(k, []):
            st = int(r["answer_start"])
            ans = r["answer"]
            assert chosen[k][st:st + len(ans)] == ans
            ds = offs[pi] + st
            qs.append({"q": r["question"], "a": ans, "s": ds, "e": ds + len(ans)})
    docs[t] = {"text": text, "qs": qs}

rng = np.random.default_rng(0)
allq = [(t, i) for t in dev_titles for i in range(len(docs[t]["qs"]))]
keep = sorted(rng.choice(len(allq), min(500, len(allq)), replace=False))
QUESTIONS = [(docs[t]["qs"][i], t) for t, i in (allq[k] for k in keep)]
print("dev questions used:", len(QUESTIONS), flush=True)

WORD_RE = re.compile(r"\S+")


def words(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def fixed_words(text, size, ov=0.0):
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


def tokenize(t):
    strip = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\u06d4\u061f\u060c\u061b'
    return [w.strip(strip) for w in t.split() if w.strip(strip)]


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
        return np.argsort(-s)[:k]


def ev(ranked, spans, cdocs, cwords, qdoc, gold, budget=500):
    used, hit, flags = 0, False, []
    for cid in ranked:
        r = cdocs[cid] == qdoc and spans[cid][0] <= gold[0] and spans[cid][1] >= gold[1]
        flags.append(r)
        if used < budget:
            used += cwords[cid]
            hit = hit or r
    flags = np.array(flags)
    r1 = float(flags[:1].any()) if len(flags) else 0.0
    r10 = float(flags[:10].any()) if len(flags) else 0.0
    mrr = 1.0 / (np.flatnonzero(flags[:10])[0] + 1) if flags[:10].any() else 0.0
    return r1, r10, mrr, float(hit), float(not flags.any())


from sentence_transformers import SentenceTransformer

print("loading bge-m3", flush=True)
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
model.max_seq_length = 512
model.half()
print("model ready", flush=True)

results = []
t0 = time.time()
for size in [50, 100, 200]:
    texts, spans, cdocs = [], [], []
    for ai, t in enumerate(dev_titles):
        for s, e in fixed_words(docs[t]["text"], size):
            texts.append(docs[t]["text"][s:e])
            spans.append((s, e))
            cdocs.append(ai)
    cwords = [len(x.split()) for x in texts]
    index = BM25([tokenize(x) for x in texts])
    bm = []
    for q, t in QUESTIONS:
        ids = index.topk(tokenize(q["q"]), 100)
        bm.append(ev(ids, spans, cdocs, cwords, dev_titles.index(t), (q["s"], q["e"])))
    te = time.time()
    emb = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    qemb = model.encode([q["q"] for q, _ in QUESTIONS], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    scores = qemb @ emb.T
    dn = []
    for qi, (q, t) in enumerate(QUESTIONS):
        ids = np.argsort(-scores[qi])[:100]
        dn.append(ev(ids, spans, cdocs, cwords, dev_titles.index(t), (q["s"], q["e"])))
    row = {"size": size, "chunks": len(texts), "embed_sec": round(time.time() - te, 1)}
    for name, arr in [("bm25", bm), ("dense", dn)]:
        a = np.array(arr)
        row[name] = {"r1": round(float(a[:, 0].mean()), 4), "r10": round(float(a[:, 1].mean()), 4),
                     "mrr": round(float(a[:, 2].mean()), 4), "bm500": round(float(a[:, 3].mean()), 4),
                     "split": round(float(a[:, 4].mean()), 4)}
    results.append(row)
    print(f"size={size} chunks={len(texts)} bm500_bm25={row['bm25']['bm500']:.3f} bm500_dense={row['dense']['bm500']:.3f} embed={row['embed_sec']}s", flush=True)

out = {"model": "BAAI/bge-m3", "split": "dev", "n_articles": len(dev_titles),
       "n_questions": len(QUESTIONS), "results": results, "total_sec": round(time.time() - t0, 1)}
json.dump(out, open("stage1_bge_m3.json", "w"), indent=2)
print("=== STAGE1_JSON_BEGIN ===")
print(json.dumps(out, separators=(",", ":")))
print("=== STAGE1_JSON_END ===")
