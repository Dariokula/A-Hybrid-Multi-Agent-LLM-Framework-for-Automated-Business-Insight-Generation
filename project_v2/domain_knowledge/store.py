# domain_knowledge/store.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os, re, json, hashlib
from pathlib import Path

from domain_knowledge.loader_pdf import extract_pdf_text
from domain_knowledge.loader_text import load_text_file
from domain_knowledge.chunking import chunk_text


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _tokenize(s: str) -> List[str]:
    import re
    s = (s or "").lower()

    # normalize odd PDF separators (common in extracted text)
    s = s.replace("\uFFFE", " ").replace("\uFFFD", " ")
    s = s.replace("￾", " ").replace("\u00AD", "")  # soft-hyphen
    s = re.sub(r"[\-/]", " ", s)

    toks = re.findall(r"[a-z0-9_äöüß]+", s)
    out: List[str] = []
    for t in toks:
        out.append(t)
        # split underscore tokens (ist_dlz -> ist, dlz)
        if "_" in t:
            out.extend([p for p in t.split("_") if p])
    return out

@dataclass
class DKChunk:
    source_id: str
    source_path: str
    source_type: str  # "pdf" | "txt" | "jsonl"
    chunk_id: str
    text: str


class DomainKnowledgeStore:
    """
    Loads all sources under domain_knowledge/sources/* and provides lightweight search.
    Uses cache/extracted to avoid re-extracting pdf text every run.
    """

    def __init__(self, root_dir: str = "domain_knowledge"):
        self.root_dir = Path(root_dir)
        self.sources_dir = self.root_dir / "sources"
        self.cache_extracted = self.root_dir / "cache" / "extracted"
        self.cache_extracted.mkdir(parents=True, exist_ok=True)

        self._chunks: List[DKChunk] = []
        self._loaded: bool = False

    def load_all(self, *, max_chunks_total: int = 5000) -> None:
        if self._loaded:
            return

        chunks: List[DKChunk] = []

        # PDFs
        pdf_dir = self.sources_dir / "pdf"
        if pdf_dir.exists():
            for p in sorted(pdf_dir.glob("*.pdf")):
                chunks.extend(self._load_pdf(str(p)))

        # Notes (txt/md)
        notes_dir = self.sources_dir / "notes"
        if notes_dir.exists():
            for p in sorted(list(notes_dir.glob("*.txt")) + list(notes_dir.glob("*.md"))):
                chunks.extend(self._load_text(str(p)))

        # Runs + Feedback (jsonl) – optional, for later
        runs_dir = self.sources_dir / "runs"
        if runs_dir.exists():
            for p in sorted(runs_dir.glob("*.jsonl")):
                chunks.extend(self._load_jsonl(str(p), source_type="runs"))

        fb_dir = self.sources_dir / "feedback"
        if fb_dir.exists():
            for p in sorted(fb_dir.glob("*.jsonl")):
                chunks.extend(self._load_jsonl(str(p), source_type="feedback"))

        # cap
        self._chunks = chunks[:max_chunks_total]
        self._loaded = True

    def _load_pdf(self, path: str) -> List[DKChunk]:
        fid = _file_hash(path)
        cache_path = self.cache_extracted / f"{fid}.json"

        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            text = data.get("text", "")
        else:
            text = extract_pdf_text(path)
            cache_path.write_text(json.dumps({"path": path, "text": text}, ensure_ascii=False), encoding="utf-8")

        # chunk
        out: List[DKChunk] = []
        for i, ch in enumerate(chunk_text(text, chunk_size=1200, overlap=150)):
            out.append(DKChunk(
                source_id=fid,
                source_path=path,
                source_type="pdf",
                chunk_id=f"{fid}:{i}",
                text=ch
            ))
        return out

    def _load_text(self, path: str) -> List[DKChunk]:
        fid = _file_hash(path)
        text = load_text_file(path)
        out: List[DKChunk] = []
        for i, ch in enumerate(chunk_text(text, chunk_size=1200, overlap=150)):
            out.append(DKChunk(fid, path, "txt", f"{fid}:{i}", ch))
        return out

    def _load_jsonl(self, path: str, source_type: str) -> List[DKChunk]:
        fid = _file_hash(path)
        out: List[DKChunk] = []
        # each line becomes one chunk (or chunk the concatenation later)
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                out.append(DKChunk(fid, path, source_type, f"{fid}:{i}", line))
        return out

    def search(self, query: str, *, top_k: int = 5, min_score: float = 0.1) -> Dict[str, Any]:
        """
        Lightweight scoring: token overlap + bonus for phrase matches.
        """
        self.load_all()

        q_tokens = _tokenize(query)
        if not q_tokens:
            return {"snippets": [], "debug": {"selected": 0, "reason": "empty query"}}

        q_set = set(q_tokens)

        scored: List[Tuple[float, DKChunk]] = []
        q_lower = query.lower()

        for ch in self._chunks:
            t = ch.text
            t_tokens = _tokenize(t)
            if not t_tokens:
                continue
            t_set = set(t_tokens)

            overlap = len(q_set & t_set)
            if overlap == 0:
                continue

            # normalize overlap a bit
            score = overlap / max(6, len(q_set))

            # phrase bonus
            if q_lower in t.lower():
                score += 0.5

            scored.append((score, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [(s, c) for s, c in scored if s >= min_score][:top_k]

        snippets = []
        for s, c in best:
            snippets.append({
                "score": float(s),
                "text": c.text,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "chunk_id": c.chunk_id,
            })

        return {
            "snippets": snippets,
            "debug": {
                "selected": len(snippets),
                "scanned_chunks": len(self._chunks),
                "query_tokens": q_tokens[:25],
                "reason": "keyword_overlap",
            }
        }