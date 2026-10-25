import json
import urllib.request

HDR = {"User-Agent": "phase3-recon/1.0"}
for name in ["Mavkif/urdu-msmarco-dataset", "uqa/UQA", "sameearif/UQA"]:
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request("https://huggingface.co/api/datasets/" + name, headers=HDR), timeout=60).read())
        print("##", name)
        print("cardData:", json.dumps(d.get("cardData"), ensure_ascii=False)[:600])
        for s in d.get("siblings", [])[:40]:
            print("  ", s["rfilename"])
        print("downloads:", d.get("downloads"))
    except Exception as exc:
        print("##", name, "ERR", exc)
