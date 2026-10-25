import csv
import html
import json
import os

POOL = json.load(open("data/processed/native_articles.json", encoding="utf-8"))
OUT = "docs/annotation"
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "article_pool.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["article_id", "title", "url", "revid", "timestamp", "words", "paragraphs"])
    for a in POOL:
        w.writerow([a["pageid"], a["title"], a["url"], a["revid"], a["timestamp"], a["words"], len(a["paragraphs"])])

with open(os.path.join(OUT, "native_worksheet.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "article_id", "article_title", "question", "answer", "paragraph_hint", "notes"])
    n = 0
    for a in POOL:
        for slot in range(1, 6):
            n += 1
            w.writerow([f"n{n:04d}", a["pageid"], a["title"], "", "", "", ""])
print("worksheet rows", n)

parts = [
    "<!doctype html><html lang='ur'><head><meta charset='utf-8'>",
    "<title>Native Urdu annotation reader</title>",
    "<style>",
    "body{font-family:system-ui,Segoe UI,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#1a1a1a;background:#faf9f7}",
    "h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2.2rem;border-top:1px solid #ddd;padding-top:1rem}",
    ".meta{color:#666;font-size:.85rem;direction:ltr;text-align:left}",
    ".article{direction:rtl;text-align:right;font-size:1.15rem;line-height:2}",
    ".article p{margin:.6rem 0}",
    ".pnum{color:#999;font-size:.7rem;direction:ltr;unicode-bidi:embed;margin-left:.4rem}",
    "nav a{display:block;padding:.15rem 0;color:#244}",
    "</style></head><body>",
    "<h1>Native Urdu annotation reader</h1>",
    "<p class='meta'>Article text pinned to the revision IDs below. Paragraph numbers are for the "
    "<code>paragraph_hint</code> column of the worksheet. Do not edit the text.</p>",
    "<nav><strong>Articles</strong>",
]
for i, a in enumerate(POOL, 1):
    parts.append(f"<a href='#a{i}'>{html.escape(a['title'])}</a>")
parts.append("</nav>")
for i, a in enumerate(POOL, 1):
    parts.append(f"<h2 id='a{i}'>{html.escape(a['title'])}</h2>")
    parts.append(f"<div class='meta'>id {a['pageid']} · rev {a['revid']} · {a['words']} words · "
                 f"{len(a['paragraphs'])} paragraphs · <a href='{html.escape(a['url'])}'>{html.escape(a['url'])}</a></div>")
    parts.append("<div class='article'>")
    for pi, p in enumerate(a["paragraphs"], 1):
        parts.append(f"<p><span class='pnum'>[{pi}]</span>{html.escape(p)}</p>")
    parts.append("</div>")
parts.append("</body></html>")
open(os.path.join(OUT, "native_reader.html"), "w", encoding="utf-8").write("\n".join(parts))

template = {
    "item_id": "n0001",
    "article_id": 0,
    "article_title": "EXACT TITLE FROM article_pool.csv",
    "question": "A self-contained Urdu question ending in ؟",
    "answer": "An exact contiguous substring of the article text",
    "paragraph_hint": 0,
    "notes": "",
}
open(os.path.join(OUT, "native_template.jsonl"), "w", encoding="utf-8").write(json.dumps(template, ensure_ascii=False) + "\n")
print("ok")
