import os
import urllib.request

HDR = {"User-Agent": "phase1-recon/1.0"}
OUT = os.path.join("data", "raw", "uqa")
os.makedirs(OUT, exist_ok=True)

jobs = [
    ("https://huggingface.co/datasets/uqa/UQA/resolve/main/data/train-00000-of-00001-bac007e8ca719235.parquet",
     "uqa_train.parquet"),
    ("https://huggingface.co/datasets/uqa/UQA/resolve/main/data/validation-00000-of-00001-cf8a6960dcbb53ee.parquet",
     "uqa_validation.parquet"),
    ("https://raw.githubusercontent.com/sameearif/UQA/main/SQuAD/dev-v2.0.json", "squad_dev.json"),
    ("https://raw.githubusercontent.com/sameearif/UQA/main/SQuAD/train-v2.0.json", "squad_train.json"),
]

for url, name in jobs:
    path = os.path.join(OUT, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("skip", name, os.path.getsize(path))
        continue
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=600) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print("got", name, os.path.getsize(path))
