import csv
import json
import os

D = "data/external/urdu_chunking_full"
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = list(csv.DictReader(f))
corpus = {}
for line in open(os.path.join(D, "corpus_ur.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    corpus[int(r["article_id"])] = r

cands = []
for q in QS:
    if q["split"] != "test":
        continue
    ans = q["answer"].strip()
    if not (4 <= len(ans) <= 30):
        continue
    if not (30 <= len(q["question"]) <= 110):
        continue
    if any(c in ans for c in "؟۔"):
        continue
    cands.append(q)

# pick a stable, readable one: first with a short answer and context under 3000 chars
for q in cands:
    art = corpus[int(q["article_id"])]
    st, en = int(q["answer_start"]), int(q["answer_end"])
    doc = art["doc"]
    if en - st != len(q["answer"]):
        continue
    snippet = doc[max(0, st - 160):min(len(doc), en + 160)]
    if len(snippet) > 700:
        continue
    out = {
        "qid": q["qid"], "title": art["title"], "article_id": q["article_id"],
        "question": q["question"], "answer": q["answer"],
        "start": st, "end": en, "paragraph": q["paragraph"],
        "snippet": snippet, "snippet_start": max(0, st - 160),
        "snippet_end": min(len(doc), en + 160),
    }
    os.makedirs("guide", exist_ok=True)
    json.dump(out, open("guide/example_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("chosen", q["qid"], "len", len(q["answer"]), "title", art["title"])
    print("question", q["question"])
    print("answer", q["answer"])
    break
