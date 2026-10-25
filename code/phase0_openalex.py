import json
import re
import sys
import urllib.parse
import urllib.request

HDR = {"User-Agent": "phase0-lit/1.0 (mailto:local@example.org)"}
q = " ".join(sys.argv[1:]) or "mMARCO multilingual MS MARCO passage ranking"
url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
    {"search": q, "per-page": 10, "select": "id,doi,title,publication_year,type,primary_location,locations,authorships"}
)
d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90))
lines = [f"QUERY: {q} | count: {d['meta']['count']}"]
for w in d["results"]:
    auth = ", ".join(a["author"]["display_name"] for a in w["authorships"][:6])
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    locs = [l.get("landing_page_url") for l in (w.get("locations") or []) if l.get("landing_page_url")]
    arx = [x for x in locs if "arxiv.org" in x]
    lines.append(f"- {w['publication_year']} | {w['title']} | {auth} | {src} | {w.get('doi')} | {arx[:2]}")
slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:60]
path = f"results/phase0/oa_{slug}.md"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", path, len(lines) - 1, "hits")
