import json
import os
import urllib.request

HDR = {"User-Agent": "phase1-recon/1.0"}
OUT = os.path.join("data", "raw", "tokenizers")
os.makedirs(OUT, exist_ok=True)

repos = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "labse": "sentence-transformers/LaBSE",
    "mpnet-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

report = []
for key, repo in repos.items():
    path = os.path.join(OUT, key + ".json")
    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
    try:
        if not os.path.exists(path):
            data = urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=180).read()
            open(path, "wb").write(data)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(path)
        vocab = tok.get_vocab_size()
        report.append(f"{key:22s} {repo:62s} bytes={os.path.getsize(path):>9d} vocab={vocab:>6d}")
    except Exception as exc:
        report.append(f"{key:22s} {repo:62s} FAILED {exc}")

open("results/phase1/tokenizer_files.txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
print("\n".join(report))
