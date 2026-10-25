from tokenizers import Tokenizer

sentence = "پاکستان کا دارالحکومت اسلام آباد ہے۔ یہ شہر 1960ء میں تعمیر ہوا۔"
word = "دارالحکومت"
lines = []
for name in ["bge-m3", "qwen3-0.6b"]:
    tok = Tokenizer.from_file(f"data/raw/tokenizers/{name}.json")
    tok.no_truncation()
    enc = tok.encode(sentence)
    we = tok.encode(word)
    lines.append(f"## {name}")
    lines.append(f"sentence tokens {len(enc.ids)} words {len(sentence.split())}")
    lines.append("word tokens " + str(len(we.ids)) + " " + " | ".join(t.replace("\u2581", "|") for t in we.tokens))
    lines.append("all tokens " + " | ".join(t.replace("\u2581", "|") for t in enc.tokens))
open("results/analysis/urdu_example_tokens.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok")
