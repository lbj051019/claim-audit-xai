from __future__ import annotations
def clean_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, size: int = 800, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks
