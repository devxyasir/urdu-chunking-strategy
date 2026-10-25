import json

src = "urdu_chunking_experiments.ipynb"
dst = "notebooks/urdu_chunking_fullrun.ipynb"
nb = json.load(open(src, encoding="utf-8"))
cell = nb["cells"][1]
text = "".join(cell["source"])
text = text.replace("DRY_RUN = True", "DRY_RUN = False")
text = text.replace("EN_BOUNDARIES_FOR_EN_CORPUS = False", "EN_BOUNDARIES_FOR_EN_CORPUS = True")
cell["source"] = text.splitlines(keepends=True)
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", dst)
print(text.splitlines()[0:10])
