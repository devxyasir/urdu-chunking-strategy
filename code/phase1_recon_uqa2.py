import json
import urllib.request

HDR = {"User-Agent": "phase1-recon/1.0"}
out = []
for path in ["UQA/train", "UQA/validation", "SQuAD-UR/train-v2.0", "SQuAD-UR/dev-v2.0"]:
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"https://api.github.com/repos/sameearif/UQA/contents/{path}", headers=HDR),
            timeout=90).read())
        out.append(f"## {path} ({len(d)})")
        for x in d:
            out.append(f"  {x['size']:>10d} {x['name']}  {x.get('download_url')}")
    except Exception as exc:
        out.append(f"## {path} ERR {exc}")
open("results/phase1/recon_uqa2.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok")
