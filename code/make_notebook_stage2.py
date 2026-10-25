import json
import os

code = open("code/colab_stage2.py", encoding="utf-8").read()

md = """# Urdu chunking — Stage 2 (full dense grid, bge-m3, test split)

Run with a **T4 GPU** runtime. This is the main run for one model.

- 120 test articles, 7,362 questions, **21 chunk configurations** (word sizes 50/100/200 ×
  overlaps 0/10/25; token sizes 64/128/256; sentence-aware, recursive-English, recursive-Urdu).
- Resumable: results are checkpointed after every configuration to
  `urdu_chunking/stage2_bge_m3.partial.json` (Google Drive if mounted, else `/content`).
  If Colab disconnects, rerun the notebook and it skips finished configurations.
- Expect **45–80 minutes**. The last cell downloads `stage2_bge_m3.json`.

**When it finishes, upload `stage2_bge_m3.json` to the workspace** (the file has per-article
data I need), and also paste the compact JSON between the STAGE2 markers if it is visible.
"""

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["!pip -q install -U sentence-transformers\n",
                "!mkdir -p /content/urdu_chunking\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "except Exception as e:\n",
                "    print('Drive not mounted, using /content:', e)\n"]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["%%writefile /content/urdu_chunking/stage2.py\n"] + code.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ["!python /content/urdu_chunking/stage2.py\n"]},
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
with open("notebooks/urdu_chunking_stage2_bge_m3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote notebooks/urdu_chunking_stage2_bge_m3.ipynb", len(code), "chars")
