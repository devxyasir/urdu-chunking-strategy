import os

import pymupdf

os.makedirs("paper/pages", exist_ok=True)
doc = pymupdf.open("paper/paper.pdf")
print("pages", doc.page_count)
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=110)
    pix.save(f"paper/pages/page{i+1}.png")
    text = page.get_text()
    refs = text.find("References")
    print(i + 1, "chars", len(text), "refs_at", refs)
