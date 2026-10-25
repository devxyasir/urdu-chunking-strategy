import re

src = ""
for f in ["refs/phase0_arxiv.bib", "refs/phase0_acl.bib", "refs/phase0_elra.bib"]:
    src += open(f, encoding="utf-8").read().strip() + "\n\n"

blocks = re.findall(r"@\w+\{[^@]+?\n\}", src, re.S)
out = []
for b in blocks:
    if b.startswith("@misc"):
        b = re.sub(r"\n\s*note = \{[^\n]*\},", "", b)
    out.append(b.strip())
open("paper/refs.bib", "w", encoding="utf-8").write("\n\n".join(out) + "\n")
print("entries", len(out), "urls", sum(1 for b in out if "http" in b))
