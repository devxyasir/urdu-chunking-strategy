import hashlib
import json
import os
from collections import Counter, defaultdict

import pyarrow.parquet as pq

os.makedirs("data/processed", exist_ok=True)
log = open("results/phase1/build_log.txt", "w", encoding="utf-8")


def say(msg):
    log.write(msg + "\n")
    log.flush()


say("load squad")
squad = json.load(open("data/raw/uqa/squad_train.json", encoding="utf-8"))
id2key, eng, order = {}, {}, {}
for art in squad["data"]:
    t = art["title"]
    order[t] = []
    for pi, p in enumerate(art["paragraphs"]):
        key = (t, pi)
        eng[key] = p["context"]
        order[t].append(key)
        for qa in p["qas"]:
            id2key[qa["id"]] = key
say(f"paragraphs {len(eng)} articles {len(order)}")

say("load uqa")
rows = pq.read_table("data/raw/uqa/uqa_train.parquet").to_pylist()
say(f"rows {len(rows)}")

say("group")
groups = defaultdict(list)
missing = 0
for r in rows:
    key = id2key.get(r["id"])
    if key is None:
        missing += 1
        continue
    groups[key].append((r["id"], r["context"], int(r["answer_start"]), r["answer"], bool(r["is_impossible"])))
say(f"groups {len(groups)} missing {missing}")

say("choose variants")
chosen, retained, variant_counts = {}, [], []
for key, rs in groups.items():
    cnt = Counter(x[1] for x in rs)
    best = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]
    chosen[key] = best
    variant_counts.append(len(cnt))
    for qid, ctx, st, ans, imp in rs:
        if ctx == best and not imp:
            retained.append({"id": qid, "key": key, "answer": ans, "answer_start": st})
say(f"retained {len(retained)}")

say("article sizes")
words_by_title, q_by_title, paras_by_title = defaultdict(int), defaultdict(int), defaultdict(int)
for key, ctx in chosen.items():
    words_by_title[key[0]] += len(ctx.split())
for r in retained:
    q_by_title[r["key"][0]] += 1
for key in chosen:
    paras_by_title[key[0]] += 1
eng_words = {t: sum(len(eng[k].split()) for k in keys) for t, keys in order.items()}

cand = [t for t in order if q_by_title[t] >= 20 and paras_by_title[t] >= 20 and words_by_title[t] >= 1500]
cand.sort(key=lambda t: hashlib.sha1(t.encode()).hexdigest())
pool = cand[:150]
split = {t: ("dev" if i % 5 == 0 else "test") for i, t in enumerate(pool)}

pool_out = {
    "articles": [
        {"title": t, "split": split[t], "paragraphs": paras_by_title[t],
         "urdu_words": words_by_title[t], "english_words": eng_words[t], "questions": q_by_title[t]}
        for t in pool
    ],
    "totals": {"articles": len(pool), "urdu_words": sum(words_by_title[t] for t in pool),
               "english_words": sum(eng_words[t] for t in pool),
               "questions": sum(q_by_title[t] for t in pool)},
}
json.dump(pool_out, open("data/processed/pool_articles.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

vs = sorted(variant_counts)
med = vs[len(vs) // 2]
all_words = sorted(words_by_title.values())
med_w = all_words[len(all_words) // 2]
p90_w = all_words[int(len(all_words) * 0.9)]
pool_words = sum(words_by_title[t] for t in pool)
pool_questions = sum(q_by_title[t] for t in pool)
dev_q = sum(q_by_title[t] for t in pool if split[t] == "dev")

lines = [
    "# Corpus construction feasibility (UQA train, measured)",
    "",
    f"- English paragraphs: {len(eng):,} across {len(order):,} articles",
    f"- UQA rows mapped to an English paragraph: {len(rows):,} (unmatched: {missing})",
    f"- distinct Urdu context variants per paragraph: mean {len(rows) / len(groups):.2f} rows, median {med} variants, max {max(variant_counts)}",
    f"- answerable questions retained under the mode-variant rule: {len(retained):,} of {len(rows):,} ({len(retained) / len(rows):.2%})",
    "",
    "## Deduplicated documents (all 441 articles)",
    f"- Urdu words per article: median {med_w:,}, p90 {p90_w:,}, max {max(all_words):,}",
    f"- total Urdu words (deduplicated): {sum(words_by_title.values()):,}",
    "",
    "## Article pool (150 articles, deterministic hash order)",
    f"- eligible articles: {len(cand):,} (>=20 retained questions, >=20 paragraphs, >=1500 Urdu words)",
    f"- pool Urdu words: {pool_words:,}; English words: {sum(eng_words[t] for t in pool):,}",
    f"- pool paragraphs: {sum(paras_by_title[t] for t in pool):,}",
    f"- pool retained answerable questions: {pool_questions:,}",
    f"- dev articles: {sum(1 for t in pool if split[t] == 'dev')}; test articles: {sum(1 for t in pool if split[t] == 'test')}",
    f"- dev retained questions: {dev_q:,}; test: {pool_questions - dev_q:,}",
    "",
    "| size (words) | chunks, overlap 0 | chunks, overlap 10% | chunks, overlap 25% |",
    "| --- | --- | --- | --- |",
]
for s in [50, 100, 200, 400, 800]:
    lines.append(f"| {s} | {pool_words // s:,} | {int(pool_words / (s * 0.9)):,} | {int(pool_words / (s * 0.75)):,}")

open("results/phase1/pool_stats.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
say(f"done pool={len(pool)} q={pool_questions} words={pool_words}")
