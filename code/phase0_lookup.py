import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
TITLES = [
    "Dense X Retrieval: What Retrieval Granularity Should We Use",
    "MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages",
    "mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset",
    "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation",
    "Multilingual E5 Text Embeddings: A Technical Report",
    "Language-agnostic BERT Sentence Embedding",
    "Searching for Best Practices in Retrieval-Augmented Generation",
    "Urdu text normalization",
    "A Survey on Urdu NLP",
]


def q(title):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": 'ti:"%s"' % title, "max_results": 4}
    )
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "phase0/1.0"}), timeout=60) as r:
        root = ET.fromstring(r.read())
    print("##", title)
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
        t = " ".join(e.find("a:title", NS).text.split())
        au = ", ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)[:3])
        print("  ", aid, "|", t[:100], "|", au)


for t in TITLES:
    try:
        q(t)
    except Exception as exc:
        print("ERR", t, exc)
