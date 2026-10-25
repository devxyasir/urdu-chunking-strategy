import os
import urllib.request

OUT = os.path.join("data", "raw", "uqa")
HDR = {"User-Agent": "phase1-recon/1.0"}
url = "https://raw.githubusercontent.com/sameearif/UQA/main/UQA/validation/data-00000-of-00001.arrow"
path = os.path.join(OUT, "validation.arrow")
with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=300) as r, open(path, "wb") as f:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
print("bytes", os.path.getsize(path))
print(open(path, "rb").read()[:120])
