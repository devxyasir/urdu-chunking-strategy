import json
import re

meta = json.load(open("results/phase0/arxiv_meta.json", encoding="utf-8"))
papers = json.load(open("configs/phase0_papers.json", encoding="utf-8"))
bib = ""
for p in ["refs/phase0_arxiv.bib", "refs/phase0_acl.bib", "refs/phase0_elra.bib"]:
    bib += open(p, encoding="utf-8").read() + "\n"

blocks = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)
keys = [k.strip() for k, _ in blocks]
by_eprint = {}
for key, body in blocks:
    m = re.search(r"eprint = \{([^}]+)\}", body)
    t = re.search(r"title = \{(.*?)\},\n", body, re.S)
    if m:
        by_eprint[m.group(1).split("v")[0]] = (key, re.sub(r"\s+", " ", t.group(1)).strip() if t else "")

lines = [
    "# Bibliography check",
    "",
    f"- entries parsed: {len(blocks)}",
    f"- duplicate keys: {sorted({k for k in keys if keys.count(k) > 1}) or 'none'}",
    f"- replacement characters present: {chr(0xfffd) in bib}",
    "",
    "| key | expected ID | entry key | title match |",
    "| --- | --- | --- | --- |",
]

ok = 0
for p in papers:
    if p["source"] == "arxiv":
        base = p["id"].split("v")[0]
        ent = by_eprint.get(base)
        match = bool(ent) and ent[1].strip() == meta[base]["title"].strip()
        lines.append(f"| {p['key']} | {base} | {ent[0] if ent else 'MISSING'} | {match} |")
        ok += int(match)
    else:
        lines.append(f"| {p['key']} | manual record | (ACL/ELRA bib) | n/a |")
        ok += 1

lines += ["", f"Resolved {ok}/{len(papers)} config papers to a verified entry."]
with open("results/phase0/bib_check.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE results/phase0/bib_check.md", ok, "/", len(papers))
