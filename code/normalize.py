import unicodedata
from bisect import bisect_left, bisect_right

DIACRITICS = set(range(0x064B, 0x0653)) | {0x0670, 0x0653, 0x0654, 0x0655}
TATWEEL = 0x0640
ZWNJ, ZWJ = 0x200C, 0x200D

LETTER_MAP = {
    "\u064a": "\u06cc",
    "\u0643": "\u06a9",
    "\u0629": "\u06c3",
}
PUNCT_MAP = {
    "\u060c": ",",
    "\u061b": ";",
    "?": "\u061f",
}
DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
DIGITS.update({chr(0x06F0 + i): str(i) for i in range(10)})


def is_nfc(text):
    return unicodedata.normalize("NFC", text) == text


def normalize_with_map(text, variant):
    if variant == "N0":
        return text, list(range(len(text)))
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        o = ord(ch)
        if ch == "\n":
            out.append("\n")
            idx.append(i)
            prev_space = False
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
                prev_space = True
            continue
        prev_space = False
        if variant in ("N1", "N2") and (o in DIACRITICS or o == TATWEEL):
            continue
        if variant == "N2":
            if o in (ZWNJ, ZWJ):
                continue
            if ch in LETTER_MAP:
                ch = LETTER_MAP[ch]
            elif ch in PUNCT_MAP:
                ch = PUNCT_MAP[ch]
            elif ch in DIGITS:
                ch = DIGITS[ch]
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def normalize(text, variant):
    return normalize_with_map(text, variant)[0]


def map_span(start, end, index_map):
    if end <= start:
        return None
    ns = bisect_left(index_map, start)
    ne = bisect_right(index_map, end - 1)
    if ns >= ne:
        return None
    return ns, ne


def map_gold(doc_norm, answer, map_start, map_end, index_map, variant):
    ns, ne = map_span(map_start, map_end, index_map)
    if ns is None:
        return None, "out_of_range"
    want = normalize(answer, variant)
    got = doc_norm[ns:ne]
    if got != want:
        return None, f"mismatch:{got[:20]}!={want[:20]}"
    return (ns, ne), "ok"
