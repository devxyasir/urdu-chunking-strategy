import urllib.request

HDR = {"User-Agent": "phase3-recon/1.0"}
base = "https://huggingface.co/datasets/Mavkif/urdu-msmarco-dataset/resolve/main/"
readme = urllib.request.urlopen(urllib.request.Request(base + "README.md", headers=HDR), timeout=60).read().decode("utf-8", "replace")
open("results/phase3_readme_urdu_msmarco.md", "w", encoding="utf-8").write(readme)
print(readme[:2000])

for f in ["urdu_collection_small.tsv", "urdu_queries.dev.small.tsv"]:
    try:
        req = urllib.request.Request(base + f, headers=HDR, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f, r.headers.get("Content-Length"))
    except Exception as exc:
        print(f, "ERR", exc)
