import json
import os
from collections import Counter, defaultdict

import pyarrow.parquet as pq

OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)


def build():
    squad = json.load(open("data/raw/uqa/squad_train.json", encoding="utf-8"))
    pool = json.load(open("data/processed/pool_articles.json", encoding="utf-8"))
    pool_titles = {a["title"]: a["split"] for a in pool["articles"]}

    id2key, id2qa, eng, order = {}, {}, {}, {}
    for art in squad["data"]:
        t = art["title"]
        order[t] = []
        for pi, p in enumerate(art["paragraphs"]):
            key = (t, pi)
            order[t].append(key)
            eng[key] = p["context"]
            for qa in p["qas"]:
                id2key[qa["id"]] = key
                if qa["answers"]:
                    id2qa[qa["id"]] = (qa["answers"][0]["text"], qa["answers"][0]["answer_start"])

    rows = pq.read_table("data/raw/uqa/uqa_train.parquet").to_pylist()
    groups = defaultdict(list)
    for r in rows:
        k = id2key.get(r["id"])
        if k is not None:
            groups[k].append(r)

    chosen = {}
    for k, rs in groups.items():
        cnt = Counter(r["context"] for r in rs)
        chosen[k] = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]

    chosen_rows = defaultdict(list)
    for r in rows:
        k = id2key.get(r["id"])
        if k is not None and r["context"] == chosen[k] and not r["is_impossible"]:
            chosen_rows[k].append(r)

    def write(path, lang):
        n_art = n_q = 0
        with open(path, "w", encoding="utf-8") as f:
            for title, split in pool_titles.items():
                paras_text, pid_map = [], []
                for pi, key in enumerate(order[title]):
                    text = chosen.get(key) if lang == "ur" else eng.get(key)
                    if text is None:
                        continue
                    pid_map.append((len(paras_text), key))
                    paras_text.append(text)
                doc = "\n\n".join(paras_text)
                offsets = []
                pos = 0
                for t in paras_text:
                    offsets.append(pos)
                    pos += len(t) + 2
                questions = []
                for (local_pid, key) in pid_map:
                    for r in chosen_rows.get(key, []):
                        if lang == "ur":
                            ptext, st, ans = chosen[key], int(r["answer_start"]), r["answer"]
                        else:
                            if r["id"] not in id2qa:
                                continue
                            ans, st = id2qa[r["id"]]
                            ptext = eng[key]
                        if ptext[st:st + len(ans)] != ans:
                            raise AssertionError(f"bad span {key} {r['id']} {lang}")
                        doc_start = offsets[local_pid] + st
                        questions.append({
                            "qid": r["id"], "question": r["question"], "answer": ans,
                            "start": doc_start, "end": doc_start + len(ans),
                            "paragraph": local_pid,
                        })
                if not questions:
                    continue
                rec = {
                    "article_id": title, "split": split, "lang": lang, "title": title,
                    "paragraphs": [{"pid": i, "text": t} for i, t in enumerate(paras_text)],
                    "doc": doc,
                    "questions": sorted(questions, key=lambda q: q["qid"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_art += 1
                n_q += len(questions)
        print(path, "articles", n_art, "questions", n_q)

    write(os.path.join(OUT, "corpus_ur.jsonl"), "ur")
    write(os.path.join(OUT, "corpus_en.jsonl"), "en")


if __name__ == "__main__":
    build()
