import json

nb = json.load(open("urdu_chunking_experiments.ipynb", encoding="utf-8"))
for i in [1, 5, 16, 17, 21]:
    open(f"results/analysis/cell{i}.txt", "w", encoding="utf-8").write("".join(nb["cells"][i]["source"]))
print("ok")
