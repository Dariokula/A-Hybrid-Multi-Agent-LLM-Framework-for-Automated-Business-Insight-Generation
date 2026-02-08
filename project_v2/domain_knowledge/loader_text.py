# domain_knowledge/loader_text.py
from __future__ import annotations

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()