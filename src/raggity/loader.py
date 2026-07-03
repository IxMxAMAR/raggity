from __future__ import annotations

import fnmatch
import glob
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import Document
from .readers import SUPPORTED_EXTS, MissingDependencyError, read_file

log = logging.getLogger("raggity.loader")

# Directory names never worth indexing.  Pruned ONLY when they appear BELOW an
# include pattern's static prefix (see _expand), so a pattern deliberately
# pointing inside such a dir still works.
_DEFAULT_EXCLUDE_DIRS = {
    "AppData", "node_modules", ".git", "__pycache__", "site-packages",
    ".venv", "venv", "dist-packages", ".raggity", ".npm", ".nuget",
    ".gradle", ".cargo", ".conda",
}

_GLOB_META = ("*", "?", "[")


def _static_prefix_len(pattern: str) -> int:
    """Number of leading path parts of *pattern* before the first glob metachar.

    e.g. ``.../notes/*.md`` -> the parts up to and including ``notes``; ``**/*.txt``
    run from a root -> only the root's parts (the ``**`` segment stops it).
    """
    parts = Path(os.path.expanduser(pattern)).parts
    n = 0
    for part in parts:
        if any(m in part for m in _GLOB_META):
            break
        n += 1
    return n


def _junk_below_prefix(fp: str, prefix_len: int) -> bool:
    """True if any DIRECTORY segment of *fp* below *prefix_len* is a junk dir."""
    parts = Path(fp).parts
    for seg in parts[prefix_len:-1]:  # dirs only (exclude the filename)
        if seg in _DEFAULT_EXCLUDE_DIRS:
            return True
    return False


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


def _expand(globs: list[str], exclude: list[str] | None = None) -> list[str]:
    """Expand *globs*, pruning junk dirs below each pattern's static prefix and
    any file matching a user *exclude* glob (fnmatch against the posix path)."""
    exclude = exclude or []
    out: set[str] = set()
    for g in globs:
        pattern = os.path.expanduser(g)
        prefix_len = _static_prefix_len(g)
        for fp in glob.glob(pattern, recursive=True):
            if _junk_below_prefix(fp, prefix_len):
                continue
            if exclude and any(
                fnmatch.fnmatch(Path(fp).as_posix(), pat) for pat in exclude
            ):
                continue
            out.add(fp)
    return sorted(out)


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
    scanned: int = 0  # total supported files considered (after exclusion)


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


def scan_sources(globs: list[str], manifest: dict[str, dict],
                 exclude: list[str] | None = None) -> ScanResult:
    """Glob + stat sweep classifying files against *manifest* (v2 shape).

    Cheap: no hashing, no parsing.  A file is *unchanged* only when BOTH its
    mtime AND size exactly match the manifest entry; ANY mismatch (or a
    missing/incomplete entry, e.g. a migrated v1 entry lacking mtime) makes it
    a *candidate* for hash-based comparison in :func:`load_paths`.
    """
    result = ScanResult()
    for fp in _expand(globs, exclude):
        p = Path(fp)
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            log.info("skipping unsupported file: %s", fp)
            result.skipped_generic += 1
            continue
        posix_path = p.as_posix()
        result.present.add(posix_path)
        result.scanned += 1
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
            log.info("skipping unstattable file %s: %s", posix_path, exc)
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
            log.warning("skipping %s - missing extra '%s': %s", posix_path, exc.extra, exc)
            result.skipped_needs_extra[exc.extra] = (
                result.skipped_needs_extra.get(exc.extra, 0) + 1
            )
            continue
        except Exception as exc:  # encrypted/corrupt PDFs, perms, etc.
            log.info("skipping unreadable file %s: %s", posix_path, exc)
            result.skipped_generic += 1
            continue
        if text is None or not text.strip():
            log.info("skipping empty/no-text file: %s", posix_path)
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


def load_documents(
    globs: list[str], exclude: list[str] | None = None
) -> tuple[list[Document], dict[str, int], int]:
    """Load documents matching *globs* (back-compat wrapper, empty manifest).

    Returns:
        (docs, skipped_needs_extra, skipped_generic)
        - docs: successfully loaded Document objects
        - skipped_needs_extra: {extra_name: count} for MissingDependencyError skips
        - skipped_generic: count of other skips (corrupt/empty/unsupported)
    """
    scan = scan_sources(globs, {}, exclude)
    load = load_paths(scan.candidates, {})
    return load.docs, load.skipped_needs_extra, scan.skipped_generic + load.skipped_generic
