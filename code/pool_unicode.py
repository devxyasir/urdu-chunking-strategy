import collections
import json

c = collections.Counter()
n = 0
for line in open("data/external/urdu_chunking_full/corpus_ur.jsonl", encoding="utf-8"):
    r = json.loads(line)
    t = r["doc"]
    n += len(t)
    c.update(t)
keys = {
    "U+06D4": "\u06d4", "U+061F": "\u061f", "!": "!", ".": ".",
    "ZWNJ": "\u200c", "ZWJ": "\u200d", "tatweel": "\u0640",
    "arabic_yeh": "\u064a", "farsi_yeh": "\u06cc", "arabic_kaf": "\u0643", "keheh": "\u06a9",
    "urdu_digit0": "\u06f0", "arabic_digit0": "\u0660",
}
lines = [f"chars {n}"]
for k, ch in keys.items():
    lines.append(f"{k}: {c.get(ch, 0)}")
open("results/analysis/pool_unicode.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
