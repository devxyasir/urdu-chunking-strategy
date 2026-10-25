import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import torch

MODEL_NAME = "BAAI/bge-m3"
MODEL_SLUG = "bge_m3"
PREFIX = ""
WORK = "/content/drive/MyDrive/urdu_chunking" if os.path.isdir("/content/drive/MyDrive") else "/content/urdu_chunking"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

SQUAD_URL = "https://raw.githubusercontent.com/sameearif/UQA/main/SQuAD/train-v2.0.json"
UQA_URL = "https://huggingface.co/datasets/uqa/UQA/resolve/main/data/train-00000-of-00001-bac007e8ca719235.parquet"


def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print("downloading", path, flush=True)
        urllib.request.urlretrieve(url, path)


fetch(SQUAD_URL, "squad_train.json")
fetch(UQA_URL, "uqa_train.parquet")

import pyarrow.parquet as pq

squad = json.load(open("squad_train.json", encoding="utf-8"))
rows = pq.read_table("uqa_train.parquet").to_pylist()
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
titles = [t for t in pool if split[t] == "test"]
print("test articles:", len(titles), flush=True)

qrows = defaultdict(list)
for r in ret:
    qrows[id2key[r["id"]]].append(r)
docs = {}
for t in titles:
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
            qs.append({"qid": r["id"], "q": r["question"], "a": ans, "s": ds, "e": ds + len(ans)})
    docs[t] = {"text": text, "qs": qs}
QUESTIONS = [(docs[t]["qs"][i], t) for t in titles for i in range(len(docs[t]["qs"]))]
TITLE_IDX = {t: i for i, t in enumerate(titles)}
QIDX = [(q, TITLE_IDX[t]) for q, t in QUESTIONS]
print("questions:", len(QUESTIONS), flush=True)

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


def fixed_tokens(text, tok_len, budget, ov):
    w = words(text)
    counts = [tok_len(text[s:e]) for s, e in w]
    out, i = [], 0
    while i < len(w):
        j, used = i, 0
        while j < len(w) and (used == 0 or used + counts[j] <= budget):
            used += counts[j]
            j += 1
        out.append((w[i][0], w[j - 1][1]))
        if j >= len(w):
            break
        i += max(1, int(round((j - i) * (1 - ov))))
    return out


from sentence_transformers import SentenceTransformer

print("loading", MODEL_NAME, flush=True)
model = SentenceTransformer(MODEL_NAME, device="cuda")
model.max_seq_length = 512
try:
    model.half()
except Exception:
    pass
tok_cache = {}


def tok_len(s):
    if s not in tok_cache:
        tok_cache[s] = len(model.tokenizer.encode(s, add_special_tokens=False))
    return tok_cache[s]


CONFIGS = []
for size in [50, 100, 200]:
    for ov in [0.0, 0.10, 0.25]:
        CONFIGS.append(("fixed_words", size, ov, None))
for budget in [64, 128, 256]:
    CONFIGS.append(("fixed_tokens", budget, 0.0, budget))
for size in [50, 100, 200]:
    CONFIGS.append(("sentence_aware", size, 0.0, None))
    CONFIGS.append(("recursive_english", size, 0.0, None))
    CONFIGS.append(("recursive_urdu", size, 0.0, None))
print("configs:", len(CONFIGS), flush=True)


def build(strategy, size, ov, budget):
    texts, spans, cdocs = [], [], []
    for ai, t in enumerate(titles):
        text = docs[t]["text"]
        if strategy == "fixed_words":
            cs = fixed_words(text, size, ov)
        elif strategy == "fixed_tokens":
            cs = fixed_tokens(text, tok_len, budget, ov)
        elif strategy == "sentence_aware":
            cs = sentence_aware(text, size, ov)
        elif strategy == "recursive_english":
            cs = recursive_english(text, size, ov)
        else:
            cs = recursive_urdu(text, size, ov)
        for s, e in cs:
            texts.append(text[s:e])
            spans.append((s, e))
            cdocs.append(ai)
    return texts, spans, cdocs


def eval_config(ranked_ids, spans, cdocs, cwords, qdoc, gold, budget=500):
    used, hit, flags = 0, False, []
    for cid in ranked_ids:
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


print("embedding queries once", flush=True)
qemb = model.encode([PREFIX + q["q"] for q, _ in QIDX], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
qemb_t = torch.tensor(qemb, dtype=torch.float16, device="cuda")
del qemb

PARTIAL = f"stage2_{MODEL_SLUG}.partial.json"
results = []
done = set()
if os.path.exists(PARTIAL):
    try:
        prev = json.load(open(PARTIAL))
        results = prev.get("configs", [])
        done = {(r["strategy"], r["size"], r["overlap"], r["budget"]) for r in results}
        print("resuming, completed", len(done), flush=True)
    except Exception:
        results, done = [], set()

t_start = time.time()
for strat, size, ov, budget in CONFIGS:
    if (strat, size, ov, budget) in done:
        continue
    t0 = time.time()
    texts, spans, cdocs = build(strat, size, ov, budget)
    cwords = [len(x.split()) for x in texts]
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    emb_t = torch.tensor(emb, dtype=torch.float16, device="cuda")
    del emb
    per_q = []
    for start in range(0, len(QIDX), 256):
        block = qemb_t[start:start + 256]
        sims = block @ emb_t.T
        top = torch.topk(sims, k=min(100, sims.shape[1]), dim=1).indices.cpu().numpy()
        for j, ids in enumerate(top):
            qi = start + j
            q, ai = QIDX[qi]
            per_q.append((ai,) + eval_config(ids, spans, cdocs, cwords, ai, (q["s"], q["e"])))
    by_article = defaultdict(list)
    for row in per_q:
        by_article[row[0]].append(row[1:])
    agg = {"r1": float(np.mean([x[1] for x in per_q])),
           "r10": float(np.mean([x[2] for x in per_q])),
           "mrr": float(np.mean([x[3] for x in per_q])),
           "bm500": float(np.mean([x[4] for x in per_q])),
           "split": float(np.mean([x[5] for x in per_q]))}
    pa = {k: {"bm500": float(np.mean([r[3] for r in v])), "r10": float(np.mean([r[1] for r in v])),
              "n": len(v)} for k, v in by_article.items()}
    row = {"strategy": strat, "size": size, "overlap": ov, "budget": budget,
           "chunks": len(texts), "sec": round(time.time() - t0, 1), "metrics": agg, "per_article": pa}
    results.append(row)
    print(f"[{len(results)}/{len(CONFIGS)}] {strat} size={size} ov={ov} chunks={len(texts)} "
          f"bm500={agg['bm500']:.4f} r10={agg['r10']:.4f} sec={row['sec']:.0f}", flush=True)
    json.dump({"configs": results}, open(PARTIAL, "w"))
    del emb_t
    torch.cuda.empty_cache()

results.sort(key=lambda r: (r["strategy"], r["size"], r["overlap"], r["budget"] or 0))

out = {"model": MODEL_NAME, "split": "test", "n_articles": len(titles), "n_questions": len(QUESTIONS),
       "configs": results, "total_sec": round(time.time() - t_start, 1)}
path = f"stage2_{MODEL_SLUG}.json"
json.dump(out, open(path, "w"), indent=1)
print("=== STAGE2_JSON_BEGIN ===")
summary = {"model": MODEL_NAME, "n_questions": len(QUESTIONS), "total_sec": out["total_sec"],
           "configs": [{"strategy": r["strategy"], "size": r["size"], "overlap": r["overlap"],
                        "chunks": r["chunks"], "metrics": r["metrics"]} for r in results]}
print(json.dumps(summary, separators=(",", ":")))
print("=== STAGE2_JSON_END ===")
from google.colab import files
files.download(path)
