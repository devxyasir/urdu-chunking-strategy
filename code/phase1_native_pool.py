import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

os.makedirs("data/processed", exist_ok=True)
HDR = {"User-Agent": "phase1-annotation-pool/1.0 (research; contact: local)"}
API = "https://ur.wikipedia.org/w/api.php"


def get_json(url, tries=5):
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90) as r:
                return json.loads(r.read())
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(str(last))


# 1. top-viewed articles, fixed month
pv_url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/ur.wikipedia/all-access/2026/09/all-days")
pv = get_json(pv_url)
top = [a["article"] for a in pv["items"][0]["articles"] if a["article"] not in ("Main_Page", "Special:Search")]
titles = [t.replace("_", " ") for t in top[:500]]
LOG = open("results/phase1/native_pool_log.txt", "w", encoding="utf-8")


def log(msg):
    LOG.write(str(msg) + "\n")
    LOG.flush()


log(f"viewed titles {len(titles)}")

# 2. extracts in batches of 10
articles = []
for i in range(0, len(titles), 5):
    batch = titles[i:i + 5]
    params = {"action": "query", "prop": "extracts|revisions", "explaintext": 1, "exlimit": 10,
              "rvprop": "ids|timestamp", "titles": "|".join(batch), "redirects": 1,
              "format": "json", "formatversion": 2}
    url = API + "?" + urllib.parse.urlencode(params)
    try:
        d = get_json(url)
    except Exception as exc:
        print("batch failed", i, exc)
        continue
    for page in d.get("query", {}).get("pages", []):
        if "missing" in page or not page.get("extract"):
            continue
        text = page["extract"]
        paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 40]
        rev = (page.get("revisions") or [{}])[0]
        articles.append({
            "pageid": page["pageid"], "title": page["title"],
            "url": "https://ur.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            "revid": rev.get("revid"), "timestamp": rev.get("timestamp"),
            "words": len(text.split()), "paragraphs": paras,
        })
    print("fetched", len(articles))
    time.sleep(1.0)

def is_list_like(a):
    t = a["title"]
    banned = ["فہرست", "زمرہ", "List of", "ساسس"]
    return any(b in t for b in banned) or a["words"] < 1000 or len(a["paragraphs"]) < 5

cand = [a for a in articles if not is_list_like(a)]
cand.sort(key=lambda a: hashlib.sha1(str(a["pageid"]).encode()).hexdigest())
pool = cand[:60]

json.dump(pool, open("data/processed/native_articles.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
summary = {"candidates": len(cand), "pool": len(pool), "words": sum(a["words"] for a in pool)}
json.dump(summary, open("results/phase1/native_pool_summary.json", "w", encoding="utf-8"), indent=2)
print(summary)
