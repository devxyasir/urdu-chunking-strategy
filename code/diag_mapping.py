import csv
import json
import os
import unicodedata

D = "data/external/urdu_chunking_full"
with open(os.path.join(D, "questions.csv"), encoding="utf-8-sig", newline="") as f:
    QS = list(csv.DictReader(f))
by_qid = {q["qid"]: q for q in QS}

corpus = {}
for line in open(os.path.join(D, "corpus_ur.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    corpus[int(r["article_id"])] = r

fails = None
with open(os.path.join(D, "runs.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if (r["corpus"] == "uqa_ur" and r["normalization"] == "N1" and r["retriever"] == "bm25_ws"
                and r["strategy"] == "fixed_words" and r["size"] == 50 and r["overlap"] == 0.0):
            arr = r["per_question"]["bm500x"]
            fails = [i for i, v in enumerate(arr) if v == -1]
            print("run", r["run_id"], "fails", len(fails))
            break

test = [q for q in QS if q["split"] == "test"]
lines = ["# N1/N2 mapping failures", ""]
for i in fails[:20]:
    q = test[i]
    art = corpus[int(q["article_id"])]
    st, en = int(q["answer_start"]), int(q["answer_end"])
    ans = q["answer"]
    ctx = art["doc"]
    lines.append(f"## {q['qid']} article {q['article_id']} {art['title']}")
    lines.append(f"- answer={ans.encode('unicode_escape').decode()}")
    lines.append(f"- span={st}-{en} doc_len={len(ctx)}")
    lines.append(f"- raw match={ctx[st:en] == ans}")
    lines.append(f"- NFC doc={unicodedata.normalize('NFC', ctx) == ctx} NFC answer={unicodedata.normalize('NFC', ans) == ans}")
    lines.append(f"- context={ctx[max(0,st-40):en+40].encode('unicode_escape').decode()}")
    lines.append(f"- raw answer chars={[hex(ord(c)) for c in ans[:20]]}")
    lines.append("")

open("results/analysis/mapping_failures.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote", len(fails))
