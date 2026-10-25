import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
papers = json.load(open("configs/phase0_papers.json", encoding="utf-8"))
ids = [p["id"] for p in papers if p["source"] == "arxiv"]
out = {}
for i in range(0, len(ids), 40):
    chunk = ids[i:i + 40]
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(chunk), "max_results": len(chunk)}
    )
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "phase0/1.0"}), timeout=120) as r:
        root = ET.fromstring(r.read())
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
        base = aid.split("v")[0]
        out[base] = {
            "arxiv_id": aid,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
            "published": e.find("a:published", NS).text[:10],
            "updated": e.find("a:updated", NS).text[:10],
            "primary_category": e.find("arxiv:primary_category", {"arxiv": "http://arxiv.org/schemas/atom"}).get("term"),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "comment": " ".join((e.find("{http://arxiv.org/schemas/atom}comment").text or "").split())
            if e.find("{http://arxiv.org/schemas/atom}comment") is not None else "",
            "journal_ref": " ".join((e.find("{http://arxiv.org/schemas/atom}journal_ref").text or "").split())
            if e.find("{http://arxiv.org/schemas/atom}journal_ref") is not None else "",
        }

with open(os.path.join("results", "phase0", "arxiv_meta.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

papers_keys = {p["key"]: p for p in papers if p["source"] == "arxiv"}
missing = [k for k, p in papers_keys.items() if p["id"].split("v")[0] not in out]
print("metadata entries:", len(out), "missing:", missing)
for k, p in papers_keys.items():
    m = out.get(p["id"].split("v")[0])
    print(f"{k:24s} {m['arxiv_id'] if m else 'MISSING':14s} {m['title'][:80] if m else ''}")
