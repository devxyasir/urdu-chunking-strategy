import gzip
import os
import shutil

src = "data/external/urdu_chunking_full/runs.jsonl"
dst = src + ".gz"
if not os.path.exists(dst):
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)
print("runs.jsonl.gz", round(os.path.getsize(dst) / 1e6, 1), "MB")
