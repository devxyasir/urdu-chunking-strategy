import argparse
import csv
import json
import os
import random
import re
import sys

POOL_PATH = "data/processed/native_articles.json"


def load_articles(path=POOL_PATH):
    return {a["pageid"]: a for a in json.load(open(path, encoding="utf-8"))}


def read_items(path):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("question") or "").strip():
                rows.append(r)
    return rows


def article_text(a):
    return "\n\n".join(a["paragraphs"])


def validate(items, articles):
    ok, bad = [], []
    seen_q = {}
    for idx, it in enumerate(items):
        item_id = it.get("item_id") or f"row{idx}"
        errs, warns = [], []
        try:
            aid = int(it["article_id"])
        except Exception:
            bad.append((item_id, ["article_id is not an integer"]))
            continue
        a = articles.get(aid)
        if a is None:
            bad.append((item_id, ["article_id not in the frozen pool"]))
            continue
        q = (it.get("question") or "").strip()
        ans = (it.get("answer") or "").strip()
        if not q or not ans:
            bad.append((item_id, ["empty question or answer"]))
            continue
        if len(q.split()) < 4:
            errs.append("question shorter than 4 words")
        if len(q.split()) > 40:
            errs.append("question longer than 40 words")
        if len(ans.split()) > 40:
            errs.append("answer longer than 40 words")
        if not re.search(r"[\u061f\u06d4?!]\s*$", q):
            warns.append("question does not end with ؟ or ۔")
        text = article_text(a)
        count = text.count(ans)
        if count == 0:
            errs.append("answer is not an exact substring of the article")
        elif count > 1:
            errs.append(f"answer occurs {count} times in the article; choose a unique span")
        try:
            ph = int(it.get("paragraph_hint") or 0)
        except Exception:
            ph = 0
            warns.append("paragraph_hint is not an integer")
        if 1 <= ph <= len(a["paragraphs"]):
            if ans not in a["paragraphs"][ph - 1]:
                errs.append("answer is not in the hinted paragraph")
        else:
            warns.append("paragraph_hint missing or out of range")
        if ans in q:
            warns.append("answer text appears inside the question")
        key = re.sub(r"\s+", " ", q)
        if key in seen_q:
            errs.append(f"duplicate question (also {seen_q[key]})")
        else:
            seen_q[key] = item_id
        if errs:
            bad.append((item_id, errs + ["WARN: " + w for w in warns]))
            continue
        start = text.find(ans)
        ok.append({
            "item_id": item_id,
            "article_id": aid,
            "article_title": a["title"],
            "revid": a["revid"],
            "url": a["url"],
            "question": q,
            "answer": ans,
            "answer_start": start,
            "answer_end": start + len(ans),
            "paragraph_index": ph if 1 <= ph <= len(a["paragraphs"]) else None,
            "notes": it.get("notes", ""),
            "warnings": warns,
        })
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="docs/annotation/native_worksheet.csv")
    ap.add_argument("--out", default="docs/annotation/native_validated.jsonl")
    ap.add_argument("--report", default="docs/annotation/validation_report.md")
    ap.add_argument("--sample-check", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default=POOL_PATH)
    args = ap.parse_args()

    articles = load_articles(args.pool)
    items = read_items(args.input)
    ok, bad = validate(items, articles)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# Native-set validation report", "",
             f"- items submitted: {len(items)}",
             f"- accepted: {len(ok)}",
             f"- rejected: {len(bad)}",
             f"- unique articles used: {len({r['article_id'] for r in ok})}",
             f"- questions per used article: {len(ok) / max(1, len({r['article_id'] for r in ok})):.1f}",
             ""]
    if bad:
        lines += ["## Rejected items", "", "| item | reason |", "| --- | --- |"]
        for item_id, errs in bad[:200]:
            lines.append(f"| {item_id} | {'; '.join(errs)} |")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if args.sample_check and ok:
        rng = random.Random(args.seed)
        sample = rng.sample(ok, min(args.sample_check, len(ok)))
        path = "docs/annotation/second_reader_sample.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "article_title", "question", "answer", "verdict_correct_incorrect_unsure", "comment"])
            for r in sample:
                w.writerow([r["item_id"], r["article_title"], r["question"], r["answer"], "", ""])
        lines.append(f"Second-reader sample written to {path} ({len(sample)} items, seed {args.seed}).")
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"accepted {len(ok)} rejected {len(bad)} -> {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
