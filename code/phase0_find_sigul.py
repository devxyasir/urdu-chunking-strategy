import re
import urllib.request

HDR = {"User-Agent": "phase0-lit/1.0"}
for i in range(1, 32):
    url = f"https://aclanthology.org/2026.sigul-1.{i}.bib"
    try:
        b = urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60).read().decode("utf-8", "replace")
    except Exception as exc:
        print(i, "ERR", exc)
        continue
    title = re.search(r'title = "(.*?)"', b, re.S)
    t = re.sub(r"[{}]", "", title.group(1)) if title else "?"
    if "Procrustes" in t or "Procrustes" in b:
        print("FOUND", i)
        print(b)
