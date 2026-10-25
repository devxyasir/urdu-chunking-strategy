import json

meta = json.load(open("results/phase0/arxiv_meta.json", encoding="utf-8"))
papers = json.load(open("configs/phase0_papers.json", encoding="utf-8"))
lines = []
for p in papers:
    if p["source"] != "arxiv":
        continue
    m = meta[p["id"].split("v")[0]]
    a = m["authors"]
    au = ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")
    lines.append(f"{p['key']} | {m['arxiv_id']} | {m['published']} | {au}")
    lines.append(f"  title: {m['title']}")
    lines.append(f"  comment: {m['comment']}")
    lines.append(f"  journal: {m['journal_ref']}")
with open("results/phase0/paper_table.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", len(lines))
