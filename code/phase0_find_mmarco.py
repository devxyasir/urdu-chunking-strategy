import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
for q in ['all:mmARCO', 'ti:"mMARCO"', 'all:"multilingual MS MARCO"']:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 6})
    root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), timeout=90).read())
    print("##", q)
    for e in root.findall("a:entry", NS):
        print("  ", e.find("a:id", NS).text, "|", " ".join(e.find("a:title", NS).text.split())[:110])
