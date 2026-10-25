import re
from collections import Counter

import pymupdf


def nums(path):
    t = "".join(p.get_text() for p in pymupdf.open(path))
    t = t.replace("\u2212", "-")
    return Counter(re.findall(r"\d+(?:\.\d+)?", t))


old = nums("paper/urdu_chunking_paper.pdf")
new = nums("paper/paper.pdf")
only_old = old - new
only_new = new - old
print("old unique", len(old), "new unique", len(new))
print("only old:", sorted(only_old.items())[:60])
print("only new:", sorted(only_new.items())[:60])
