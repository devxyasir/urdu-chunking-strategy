import pymupdf

d = pymupdf.open("paper/paper.pdf")
with open("paper/page5_text.txt", "w", encoding="utf-8") as f:
    f.write(d[4].get_text())
    f.write("\n\n===== WORDS IN LEFT COLUMN =====\n")
    for w in d[4].get_text("words"):
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        if x0 < 300:
            f.write(f"{y0:7.1f} {x0:6.1f} {txt}\n")
print("ok")
