import json
import urllib.request

HDR = {"User-Agent": "phase1-recon/1.0"}
out = []


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90).read()


for path in ["", "UQA", "SQuAD-UR", "SQuAD", "Annotations"]:
    url = "https://api.github.com/repos/sameearif/UQA/contents/" + path
    try:
        d = json.loads(get(url))
        out.append(f"## {path or '/'} ({len(d)})")
        for x in d:
            out.append(f"  {x['type']:4s} {x['size']:>9d} {x['name']}")
    except Exception as exc:
        out.append(f"## {path or '/'} ERR {exc}")

for url in [
    "https://huggingface.co/api/datasets/uqa",
    "https://huggingface.co/api/models/uqa",
    "https://huggingface.co/uqa",
]:
    try:
        out.append(f"## {url} -> {len(get(url))} bytes")
    except Exception as exc:
        out.append(f"## {url} ERR {exc}")

open("results/phase1/recon_uqa.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("WROTE", len(out))
