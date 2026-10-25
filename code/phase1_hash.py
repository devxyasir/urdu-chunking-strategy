import datetime
import hashlib

h = hashlib.sha256(open("prereg.md", "rb").read()).hexdigest()
with open("results/phase1/prereg_hash.txt", "w", encoding="utf-8") as f:
    f.write("prereg.md sha256 " + h + "\n")
    f.write("frozen " + datetime.datetime.now().isoformat() + "\n")
print("HASH", h)
