import re

WORD_RE = re.compile(r"\S+")


def word_spans(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def paragraph_spans(text):
    spans = []
    for m in re.finditer(r"[^\n]+", text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e))
    return spans


def _urdu_sentences(text, ps, pe):
    spans = []
    start = ps
    i = ps
    while i < pe:
        ch = text[i]
        if ch in "\u06d4\u061f!" and (i + 1 == pe or text[i + 1].isspace()):
            e = i + 1
            while e < pe and text[e].isspace():
                e += 1
            if text[start:e].strip():
                spans.append((start, i + 1))
            start = e
            i = e
        else:
            i += 1
    if start < pe and text[start:pe].strip():
        spans.append((start, pe))
    return spans


def _english_sentences(text, ps, pe):
    spans = []
    start = ps
    i = ps
    while i < pe:
        ch = text[i]
        end_ok = ch in "?!" or (
            ch == "." and (i + 2 >= pe or (text[i + 1].isspace() and (i + 2 < pe and text[i + 2].isupper())))
        )
        if end_ok:
            e = i + 1
            while e < pe and text[e].isspace():
                e += 1
            if text[start:e].strip():
                spans.append((start, i + 1))
            start = e
            i = e
        else:
            i += 1
    if start < pe and text[start:pe].strip():
        spans.append((start, pe))
    return spans


def _split_oversize(text, span, size):
    s, e = span
    ws = [(m.start(), m.end()) for m in WORD_RE.finditer(text, s, e)]
    if len(ws) <= size:
        return [span]
    out = []
    for i in range(0, len(ws), size):
        out.append((ws[i][0], ws[min(i + size, len(ws)) - 1][1]))
    return out


def _units(text, mode, size):
    units = []
    for ps, pe in paragraph_spans(text):
        if mode == "urdu":
            sents = _urdu_sentences(text, ps, pe)
        else:
            sents = _english_sentences(text, ps, pe)
        for sp in sents:
            units.extend(_split_oversize(text, sp, size))
    return units


def _pack_units(text, units, size, overlap, split_oversize=True):
    counts = [len(WORD_RE.findall(text[s:e])) for s, e in units]
    chunks = []
    i = 0
    n = len(units)
    while i < n:
        j, used = i, 0
        while j < n and (used == 0 or used + counts[j] <= size):
            used += counts[j]
            j += 1
        chunks.append((units[i][0], units[j - 1][1]))
        if j >= n:
            break
        target = size * overlap
        back, acc = 0, 0
        while j - back - 1 > i and acc < target:
            back += 1
            acc += counts[j - back]
        i = j - back
    return chunks


def chunk_fixed_words(text, size, overlap):
    words = word_spans(text)
    if not words:
        return []
    step = max(1, int(round(size * (1 - overlap))))
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + size, len(words))
        chunks.append((words[i][0], words[j - 1][1]))
        if j >= len(words):
            break
        i += step
    return chunks


def chunk_sentence_aware(text, size, overlap):
    units = _units(text, "urdu", 10 ** 9)
    return _pack_units(text, units, size, overlap)


def chunk_recursive_english(text, size, overlap):
    units = _units(text, "english", size)
    return _pack_units(text, units, size, overlap)


def chunk_recursive_urdu(text, size, overlap):
    units = _units(text, "urdu", size)
    return _pack_units(text, units, size, overlap)


def chunk_fixed_tokens(text, token_len, budget, overlap):
    words = word_spans(text)
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words):
            cand = text[words[i][0]:words[j][1]]
            if token_len(cand) > budget and j > i:
                break
            j += 1
        chunks.append((words[i][0], words[j - 1][1]))
        if j >= len(words):
            break
        k = max(1, int(round((j - i) * (1 - overlap))))
        i += k
    return chunks


CHUNKERS = {
    "fixed_words": lambda t, s, o: chunk_fixed_words(t, s, o),
    "sentence_aware": lambda t, s, o: chunk_sentence_aware(t, s, o),
    "recursive_english": lambda t, s, o: chunk_recursive_english(t, s, o),
    "recursive_urdu": lambda t, s, o: chunk_recursive_urdu(t, s, o),
}
