from __future__ import annotations

import glob
import hashlib
import logging
import os
from pathlib import Path

from .models import Document
from .readers import SUPPORTED_EXTS, read_file

log = logging.getLogger("raggity.loader")


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


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
        if ext not in SUPPORTED_EXTS:
            log.warning("skipping unsupported file: %s", fp)
            continue
        try:
            text = read_file(fp)
        except Exception as exc:  # encrypted/corrupt PDFs, perms, etc.
            log.warning("skipping unreadable file %s: %s", fp, exc)
            continue
        if text is None or not text.strip():
            log.warning("skipping empty/no-text file: %s", fp)
            continue
        posix_path = p.as_posix()  # normalize to forward slashes cross-platform
        docs.append(
            Document(
                path=posix_path,
                title=_title_for(Path(posix_path), text),
                text=text,
                file_hash=compute_file_hash(fp),
                mtime=p.stat().st_mtime,
            )
        )
    return docs
