from __future__ import annotations

import glob
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import Document
from .readers import SUPPORTED_EXTS, MissingDependencyError, read_file

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


@dataclass
class ScanResult:
    """Result of a cheap glob+stat sweep against the manifest.

    - ``unchanged``: {path: manifest_entry} for files whose (mtime, size)
      exactly match the manifest — carried forward without hashing/parsing.
    - ``candidates``: paths that are new or stat-mismatched — need hashing.
    - ``present``: every supported file that exists on disk (unchanged +
      candidates) — used for deletion detection.
    - ``skipped_generic``: unsupported-extension files encountered.
    """

    unchanged: dict[str, dict] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    present: set[str] = field(default_factory=set)
    skipped_generic: int = 0


@dataclass
class LoadResult:
    """Result of hashing + parsing the scan candidates.

    - ``docs``: successfully parsed Document objects (new/changed content).
    - ``unchanged_by_hash``: {path: fresh_manifest_entry} for files whose
      stat changed but whose content hash still matches the manifest — the
      entry carries the OLD hash with a REFRESHED (mtime, size); NOT parsed.
    - ``skipped_needs_extra``: {extra_name: count} for MissingDependencyError.
    - ``skipped_generic``: count of other skips (corrupt/empty).
    """

    docs: list[Document] = field(default_factory=list)
    unchanged_by_hash: dict[str, dict] = field(default_factory=dict)
    skipped_needs_extra: dict[str, int] = field(default_factory=dict)
    skipped_generic: int = 0


def scan_sources(globs: list[str], manifest: dict[str, dict]) -> ScanResult:
    """Glob + stat sweep classifying files against *manifest* (v2 shape).

    Cheap: no hashing, no parsing.  A file is *unchanged* only when BOTH its
    mtime AND size exactly match the manifest entry; ANY mismatch (or a
    missing/incomplete entry, e.g. a migrated v1 entry lacking mtime) makes it
    a *candidate* for hash-based comparison in :func:`load_paths`.
    """
    result = ScanResult()
    for fp in _expand(globs):
        p = Path(fp)
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            log.warning("skipping unsupported file: %s", fp)
            result.skipped_generic += 1
            continue
        posix_path = p.as_posix()
        result.present.add(posix_path)
        entry = manifest.get(posix_path)
        if entry and "mtime" in entry and "size" in entry:
            try:
                st = p.stat()
            except OSError:
                result.candidates.append(posix_path)
                continue
            if entry["mtime"] == st.st_mtime and entry["size"] == st.st_size:
                result.unchanged[posix_path] = entry
                continue
        result.candidates.append(posix_path)
    return result


def load_paths(candidates: list[str], manifest: dict[str, dict]) -> LoadResult:
    """Hash + (conditionally) parse *candidates*.

    For each candidate the content hash is computed first.  If it matches the
    manifest's stored hash the file is content-unchanged despite a stat change
    — its entry is refreshed (new mtime/size, old hash) WITHOUT parsing.
    Otherwise the file is read; MissingDependencyError / generic errors / empty
    text cause a skip that NEVER produces a manifest entry.
    """
    result = LoadResult()
    for posix_path in candidates:
        p = Path(posix_path)
        try:
            st = p.stat()
        except OSError as exc:
            log.warning("skipping unstattable file %s: %s", posix_path, exc)
            result.skipped_generic += 1
            continue
        file_hash = compute_file_hash(posix_path)
        prev = manifest.get(posix_path)
        if prev is not None and prev.get("hash") == file_hash:
            # Content identical; stat lied (touched/copied).  Refresh entry,
            # keep the old hash, do NOT re-parse.
            result.unchanged_by_hash[posix_path] = {
                "hash": file_hash,
                "mtime": st.st_mtime,
                "size": st.st_size,
            }
            continue
        try:
            text = read_file(posix_path)
        except MissingDependencyError as exc:
            log.warning("skipping %s — missing extra '%s': %s", posix_path, exc.extra, exc)
            result.skipped_needs_extra[exc.extra] = (
                result.skipped_needs_extra.get(exc.extra, 0) + 1
            )
            continue
        except Exception as exc:  # encrypted/corrupt PDFs, perms, etc.
            log.warning("skipping unreadable file %s: %s", posix_path, exc)
            result.skipped_generic += 1
            continue
        if text is None or not text.strip():
            log.warning("skipping empty/no-text file: %s", posix_path)
            result.skipped_generic += 1
            continue
        result.docs.append(
            Document(
                path=posix_path,
                title=_title_for(p, text),
                text=text,
                file_hash=file_hash,
                mtime=st.st_mtime,
                size=st.st_size,
            )
        )
    return result


def load_documents(globs: list[str]) -> tuple[list[Document], dict[str, int], int]:
    """Load documents matching *globs* (back-compat wrapper, empty manifest).

    Returns:
        (docs, skipped_needs_extra, skipped_generic)
        - docs: successfully loaded Document objects
        - skipped_needs_extra: {extra_name: count} for MissingDependencyError skips
        - skipped_generic: count of other skips (corrupt/empty/unsupported)
    """
    scan = scan_sources(globs, {})
    load = load_paths(scan.candidates, {})
    return load.docs, load.skipped_needs_extra, scan.skipped_generic + load.skipped_generic
