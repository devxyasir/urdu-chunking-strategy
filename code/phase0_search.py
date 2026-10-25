import csv
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = os.path.join("results", "phase0")
os.makedirs(OUT, exist_ok=True)

QUERIES = [
    ("chunking", 'all:"chunking" AND all:"retrieval"'),
    ("chunking", 'all:"chunk size" AND all:"RAG"'),
    ("chunking", 'ti:"chunking" AND cat:cs.IR'),
    ("chunking", 'all:"semantic chunking"'),
    ("chunking", 'all:"passage granularity"'),
    ("multilingual-retrieval", 'all:"multilingual retrieval" AND all:"dense"'),
    ("multilingual-retrieval", 'all:"cross-lingual retrieval" AND all:"embedding"'),
    ("multilingual-retrieval", 'all:"low-resource" AND all:"retrieval" AND all:"BM25"'),
    ("multilingual-retrieval", 'all:"multilingual" AND all:"BM25" AND all:"sparse"'),
    ("urdu", 'all:"Urdu" AND all:"retrieval"'),
    ("urdu", 'all:"Urdu" AND all:"question answering"'),
    ("urdu", 'all:"Urdu" AND all:"normalization"'),
    ("urdu", 'all:"Urdu" AND all:"tokenization"'),
    ("tokenizer", 'all:"tokenizer" AND all:"fertility"'),
    ("tokenizer", 'all:"tokenization" AND all:"multilingual" AND all:"fairness"'),
    ("embedding", 'all:"embedding" AND all:"multilingual" AND all:"retrieval" AND cat:cs.CL'),
    ("evaluation", 'all:"retrieval" AND all:"recall" AND all:"answer span"'),
]


def fetch(url, tries=3, pause=3.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phase0-lit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            if i == tries - 1:
                raise
            time.sleep(pause * (i + 1))


rows = {}
seen = set()
for tag, q in QUERIES:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": 30, "sortBy": "relevance"}
    )
    try:
        root = ET.fromstring(fetch(url))
    except Exception as exc:
        print("FAILED", tag, q, exc)
        continue
    n = 0
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
        base = aid.split("v")[0]
        n += 1
        if base in seen:
            continue
        seen.add(base)
        rows[base] = {
            "tag": tag,
            "arxiv_id": aid,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "authors": "; ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)),
            "published": e.find("a:published", NS).text[:10],
            "category": e.find("a:category", NS).get("term") if e.find("a:category", NS) is not None else "",
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
        }
    print(f"{tag:24s} {q[:52]:52s} results={n:2d} new_total={len(rows)}")
    time.sleep(3.0)

path = os.path.join(OUT, "candidates.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "arxiv_id", "title", "authors", "published", "category", "abstract"])
    w.writeheader()
    for k in sorted(rows, key=lambda k: rows[k]["published"], reverse=True):
        w.writerow(rows[k])
print("WROTE", path, len(rows), "candidates")
