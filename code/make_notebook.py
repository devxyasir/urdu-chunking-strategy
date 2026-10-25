import json
import os

code = open("code/colab_stage1.py", encoding="utf-8").read()

md = """# Urdu chunking — Stage 1 (pilot on the GPU)

Run this notebook in Colab with a **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU).

What it does:
1. downloads SQuAD2.0 + UQA,
2. rebuilds the exact 150-article pool and the 30 dev articles,
3. runs BM25 and bge-m3 on 500 dev questions at chunk sizes 50/100/200 words,
4. prints one compact JSON block.

Run all cells (Runtime → Run all). Expect 10–30 minutes, most of it model download and embedding.

**When it finishes, copy everything between `=== STAGE1_JSON_BEGIN ===` and `=== STAGE1_JSON_END ===`
and send it to me.** If a cell fails, copy the error text instead.
"""

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["!pip -q install -U sentence-transformers\n", "!mkdir -p /content/urdu_chunking\n"]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["%%writefile /content/urdu_chunking/stage1.py\n"] + code.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["!python /content/urdu_chunking/stage1.py\n"]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["import json\n",
                "d = json.load(open('/content/urdu_chunking/stage1_bge_m3.json'))\n",
                "print(json.dumps(d, separators=(',', ':')))\n"]},
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/urdu_chunking_stage1.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote notebooks/urdu_chunking_stage1.ipynb", len(code), "chars of code")
