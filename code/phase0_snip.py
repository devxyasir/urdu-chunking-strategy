import os
import re
import sys

key = sys.argv[1]
pat = re.compile(sys.argv[2], re.I)
t = open(os.path.join("data", "raw", "papers", key + ".txt"), encoding="utf-8", errors="replace").read()
t = re.sub(r"\s+", " ", t)
hits = []
for m in pat.finditer(t):
    start = max(0, m.start() - 120)
    s = t[start:m.start() + 220]
    if s not in hits:
        hits.append(s)
    if len(hits) >= 6:
        break
out = "\n---\n".join(hits)
with open("results/phase0/snip.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")
print("WROTE", key, len(hits))
