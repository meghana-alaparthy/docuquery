"""Split documents into overlapping, sentence-aware chunks."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def split_text(text, chunk_size=800, chunk_overlap=120):
    """Chunk text on sentence boundaries; carry a trailing overlap into the next chunk."""
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append(" ".join(current))

    for sent in sentences:
        # a single overlong sentence gets hard-split
        while len(sent) > chunk_size:
            flush()
            current.clear()
            chunks.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if current and current_len + 1 + len(sent) > chunk_size:
            flush()
            tail = chunks[-1][-chunk_overlap:]
            current = [tail] if tail.strip() else []
            current_len = len(tail)
        current.append(sent)
        current_len += (1 if current_len else 0) + len(sent)

    flush()
    return [c for c in chunks if c.strip()]
