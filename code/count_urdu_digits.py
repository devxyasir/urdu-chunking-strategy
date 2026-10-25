import json

c = {i: 0 for i in range(0x06F0, 0x06FA)}
n = 0
for line in open("data/external/urdu_chunking_full/corpus_ur.jsonl", encoding="utf-8"):
    t = json.loads(line)["doc"]
    n += len(t)
    for ch in t:
        o = ord(ch)
        if 0x06F0 <= o <= 0x06F9:
            c[o] += 1
print("chars", n, "urdu_digits_total", sum(c.values()), "by_digit", c)
