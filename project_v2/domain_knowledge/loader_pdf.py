# domain_knowledge/loader_pdf.py
from __future__ import annotations
from typing import List
from pathlib import Path

def extract_pdf_text(path: str) -> str:
    """
    Robust PDF text extractor with minimal dependencies.
    Uses PyPDF2 (commonly installed). If not installed, raise a clear error.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        import PyPDF2
    except Exception as e:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. Install via: pip install PyPDF2"
        ) from e

    text_parts: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                text_parts.append(t)

    return "\n\n".join(text_parts).strip()