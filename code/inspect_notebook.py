import json
import re

nb = json.load(open("urdu_chunking_experiments.ipynb", encoding="utf-8"))
lines = []
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    head = " | ".join(src.splitlines()[:2])[:160]
    lines.append(f"### cell {i} [{c['cell_type']}] {head}")
    if c["cell_type"] != "code":
        continue
    for pat in ["DRY_RUN", "STAGES", "MODELS", "TOKEN", "budget_mode", "N_ARTICLES", "RUN_SPLITS", "CHAR_GRID_MODE"]:
        for m in re.finditer(pat, src):
            s = max(0, m.start() - 120)
            lines.append("  >> " + src[s:m.start() + 200].replace("\n", " ")[:320])
            break
open("results/analysis/notebook_cells.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("cells", len(nb["cells"]))
