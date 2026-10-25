import os
import zipfile

RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
src = "urdu_chunking_results (1).zip"
dst = os.path.abspath("data/external/urdu_chunking_full")
os.makedirs(dst, exist_ok=True)
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/").lstrip("/")
        parts = name.split("/")
        base, ext = os.path.splitext(parts[-1])
        if base.upper() in RESERVED:
            parts[-1] = base + "_" + ext
        out = os.path.join(dst, *parts)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if info.is_dir():
            continue
        with z.open(info) as f, open(out, "wb") as g:
            g.write(f.read())
print("extracted to", dst)
