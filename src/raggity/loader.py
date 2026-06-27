from __future__ import annotations

import glob
import hashlib
import logging
import os
from pathlib import Path

from .models import Document

log = logging.getLogger("raggity.loader")

SUPPORTED = {".md", ".txt", ".pdf"}


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _title_for(path: Path, text: str) -> str:
    if path.suffix.lower() == ".md":
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return path.stem


def _expand(globs: list[str]) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(glob.glob(os.path.expanduser(g), recursive=True))
    return sorted(set(out))


def load_documents(globs: list[str]) -> list[Document]:
    docs: list[Document] = []
    for fp in _expand(globs):
        p = Path(fp)
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            log.warning("skipping unsupported file: %s", fp)
            continue
        try:
            if ext == ".pdf":
                text = read_pdf(fp)
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # encrypted/corrupt PDFs, perms, etc.
            log.warning("skipping unreadable file %s: %s", fp, exc)
            continue
        if not text.strip():
            log.warning("skipping empty/no-text file: %s", fp)
            continue
        docs.append(
            Document(
                path=fp,
                title=_title_for(p, text),
                text=text,
                file_hash=compute_file_hash(fp),
                mtime=p.stat().st_mtime,
            )
        )
    return docs
