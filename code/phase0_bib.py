import json
import re
import urllib.request

meta = json.load(open("results/phase0/arxiv_meta.json", encoding="utf-8"))
papers = json.load(open("configs/phase0_papers.json", encoding="utf-8"))


def key_for(m):
    first = re.sub(r"[^a-z]", "", m["authors"][0].split()[-1].lower())
    word = re.sub(r"[^a-z0-9]", "", m["title"].split()[0].lower())
    return f"{first}{m['published'][:4]}{word}"


entries = []
for p in papers:
    if p["source"] != "arxiv":
        continue
    m = meta[p["id"].split("v")[0]]
    note = "arXiv:" + m["arxiv_id"]
    if m["journal_ref"]:
        note += "; " + m["journal_ref"]
    if m["comment"] and ("accepted" in m["comment"].lower() or "published" in m["comment"].lower()):
        note += "; " + m["comment"]
    fields = [
        ("title", "{" + m["title"] + "}"),
        ("author", " and ".join(m["authors"])),
        ("year", m["published"][:4]),
        ("eprint", m["arxiv_id"]),
        ("archivePrefix", "arXiv"),
        ("primaryClass", m["primary_category"]),
        ("note", note),
    ]
    body = ",\n  ".join((f"title = {v}" if k == "title" else f"{k} = {{{v}}}") for k, v in fields)
    entries.append(f"@misc{{{key_for(m)},\n  {body}\n}}")

with open("refs/phase0_arxiv.bib", "w", encoding="utf-8") as f:
    f.write("\n\n".join(entries) + "\n")

HDR = {"User-Agent": "phase0-lit/1.0"}
acl = {}
for name, url in {
    "sigul2026_procrustes": "https://aclanthology.org/2026.sigul-1.23.bib",
    "umirage2026": "https://aclanthology.org/2026.abjadnlp-1.55.bib",
}.items():
    acl[name] = urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60).read().decode("utf-8")

with open("refs/phase0_acl.bib", "w", encoding="utf-8") as f:
    f.write(acl["sigul2026_procrustes"].strip() + "\n\n" + acl["umirage2026"].strip() + "\n")

with open("refs/phase0_elra.bib", "w", encoding="utf-8") as f:
    f.write(
        '@inproceedings{ahmad-etal-2026-drrag,\n'
        '  title = "{DR-RAG}: Addressing Retrieval Misalignment in Low-Resource {U}rdu Question Answering",\n'
        '  author = "Ahmad, Saad and Hammad, Muhammad and Zeeshan, Muhammad and Ullah, Faizad and Karim, Asim",\n'
        '  booktitle = "Proceedings of the Second Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2026)",\n'
        '  month = may,\n  year = "2026",\n  address = "Palma, Mallorca, Spain",\n'
        '  publisher = "ELRA Language Resources Association (ELRA)",\n'
        '  pages = "49--58",\n  doi = "10.63317/4wwyss5zkwxs",\n'
        '  url = "http://www.lrec-conf.org/proceedings/lrec2026/workshops/chipsal/pdf/2026.chipsal-1.6.pdf"\n}\n'
    )
print("WROTE refs/phase0_arxiv.bib", len(entries), "entries; acl 2; elra 1")
