import csv
import hashlib
import json
import os
import re
import time
import urllib.request

from pypdf import PdfReader

CFG = "configs/phase0_papers.json"
OUTDIR = os.path.join("data", "raw", "papers")
META = os.path.join("results", "phase0")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(META, exist_ok=True)

papers = json.load(open(CFG, encoding="utf-8"))
HDR = {"User-Agent": "phase0-lit/1.0 (academic reading; contact: local)"}


def download(url, path, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                raise ValueError("not a PDF: " + data[:20].decode("latin-1"))
            with open(path, "wb") as f:
                f.write(data)
            return data
        except Exception as exc:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


rows = []
for p in papers:
    key, src, ident = p["key"], p["source"], p["id"]
    pdf = os.path.join(OUTDIR, key + ".pdf")
    txt = os.path.join(OUTDIR, key + ".txt")
    url = f"https://arxiv.org/pdf/{ident}" if src == "arxiv" else ident
    try:
        if not os.path.exists(pdf):
            data = download(url, pdf)
        else:
            data = open(pdf, "rb").read()
        if not os.path.exists(txt):
            reader = PdfReader(pdf)
            pages = [(pg.extract_text() or "") for pg in reader.pages]
            with open(txt, "w", encoding="utf-8") as f:
                f.write("\n\n=== PAGE BREAK ===\n\n".join(pages))
        body = open(txt, encoding="utf-8").read()
        rows.append({
            "key": key,
            "cluster": p["cluster"],
            "source": src,
            "id": ident,
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest()[:16],
            "pdf_bytes": len(data),
            "pages": body.count("=== PAGE BREAK ===") + 1,
            "chars": len(body),
        })
        print(f"OK   {key:24s} pages={rows[-1]['pages']:3d} chars={len(body):7d}")
    except Exception as exc:
        rows.append({"key": key, "cluster": p["cluster"], "source": src, "id": ident, "url": url,
                     "sha256": "", "pdf_bytes": 0, "pages": 0, "chars": 0, "error": str(exc)[:120]})
        print(f"FAIL {key:24s} {exc}")

with open(os.path.join(META, "paper_index.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["key", "cluster", "source", "id", "url", "sha256", "pdf_bytes", "pages", "chars", "error"])
    w.writeheader()
    w.writerows(rows)
print("total", len(rows), "ok", sum(1 for r in rows if r["chars"] > 0))
