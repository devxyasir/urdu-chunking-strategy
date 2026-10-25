import re
import urllib.request

HDR = {"User-Agent": "phase0-lit/1.0 (research; contact: local)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90).read().decode("utf-8", "replace")


h = get("https://aclanthology.org/events/sigul-2026/")
i = h.find("Procrustes Alignment")
window = h[max(0, i - 3000):i]
print("NEAREST LINKS:", re.findall(r"/2026\.sigul-1\.\d+/", window)[-3:])

print("--- SIGUL bib ---")
print(get("https://aclanthology.org/2026.sigul-1.24.bib")[:400])

print("--- U-MIRAGE bib ---")
print(get("https://aclanthology.org/2026.abjadnlp-1.55.bib")[:1200])
