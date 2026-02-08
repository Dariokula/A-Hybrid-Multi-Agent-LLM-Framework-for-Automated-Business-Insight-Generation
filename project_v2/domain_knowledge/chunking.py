# domain_knowledge/chunking.py
from __future__ import annotations
from typing import List

def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = text or ""
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        out.append(text[i:j])
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return out