import pymupdf

for f in ["paper/urdu_chunking_paper.pdf", "paper/urdu_chunking_paper_anon.pdf"]:
    d = pymupdf.open(f)
    t = d[0].get_text()
    print(f, d.page_count, "Muhammad Yasir" in t, "Anonymous" in t)
