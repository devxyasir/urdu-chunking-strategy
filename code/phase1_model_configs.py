import json
import os
import urllib.request

HDR = {"User-Agent": "phase1-recon/1.0"}
repos = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "labse": "sentence-transformers/LaBSE",
    "mpnet-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
lines = []
for key, repo in repos.items():
    for fname in ["config.json", "sentence_bert_config.json", "tokenizer_config.json"]:
        url = f"https://huggingface.co/{repo}/raw/main/{fname}"
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90).read())
            keep = {k: d[k] for k in ["max_position_embeddings", "max_seq_length", "model_max_length", "model_type", "hidden_size"] if k in d}
            lines.append(f"{key:22s} {fname:28s} {json.dumps(keep)}")
        except Exception as exc:
            lines.append(f"{key:22s} {fname:28s} ERR {exc}")
open("results/phase1/model_configs.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
