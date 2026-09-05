"""Find document folders worth offering to index.

Metadata only: names, extensions and counts. Nothing here opens a user's
documents, and nothing here indexes anything — it produces candidates that a
human confirms.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .readers import SUPPORTED_EXTS

# Images are in SUPPORTED_EXTS but only readable with the OCR extra, so a photo
# folder would be offered as "documents" and then ingest almost nothing.
DISCOVER_EXTS: frozenset[str] = frozenset(
    e for e in SUPPORTED_EXTS
    if e not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"})


@dataclass(frozen=True)
class Candidate:
    path: Path
    kind: str          # "obsidian" | "cwd" | "known" | "found"
    file_count: int
    exts: dict[str, int]
    confidence: int    # sort key, high first
    why: str


# Counting stops here. The number exists to help someone choose between
# folders, and "5,000+" answers that as well as an exact figure would — while
# an exact figure over a large tree is what made `rag discover` take minutes.
COUNT_CAP = 5_000
_COUNT_SKIP = {"node_modules", "__pycache__", "venv", ".venv", "dist", "build",
               "site-packages", ".git", "AppData", "Library"}


def count_indexable(path: Path, *, deadline: float | None = None,
                    cap: int = COUNT_CAP) -> tuple[int, dict[str, int]]:
    """How many files under *path* ingest would take, by extension.

    Bounded on purpose. The obvious implementation is one recursive glob, and
    that is what shipped first — but `glob` walks the whole subtree, cannot be
    interrupted, and on a real `~/Documents` it turned a command budgeted at
    1.5s into minutes. This walks with `os.walk` so junk directories are pruned
    before they are descended into, and stops at *cap* or *deadline*.

    A total equal to *cap* means "at least this many"; `describe` says so.
    """
    total = 0
    exts: Counter[str] = Counter()
    for root, dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in _COUNT_SKIP]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in DISCOVER_EXTS:
                exts[ext] += 1
                total += 1
                if total >= cap:
                    return total, dict(sorted(exts.items()))
        if deadline is not None and time.monotonic() > deadline:
            break
    return total, dict(sorted(exts.items()))


def describe(exts: dict[str, int]) -> str:
    """`341 notes` / `1,204 documents - pdf, docx, md` / `5,000+ documents`."""
    total = sum(exts.values())
    if total >= COUNT_CAP:
        return f"{COUNT_CAP:,}+ documents"
    kinds = ", ".join(e.lstrip(".") for e, _ in
                      sorted(exts.items(), key=lambda kv: -kv[1])[:3])
    noun = "note" if set(exts) <= {".md", ".txt"} else "document"
    return f"{total:,} {noun}{'s' if total != 1 else ''}" + (
        f" - {kinds}" if len(exts) > 1 else "")


def scan_named(roots: list[Path],
               deadline: float | None = None) -> list[Candidate]:
    """Candidates for a fixed list of directories. Missing or empty ones are
    simply absent — an offer with nothing behind it wastes the user's read."""
    out: list[Candidate] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        total, exts = count_indexable(root, deadline=deadline)
        if not total:
            continue
        out.append(Candidate(path=root, kind="known", file_count=total,
                             exts=exts, confidence=50, why=describe(exts)))
    return out


def obsidian_config_path() -> Path | None:
    """Where Obsidian records its vault list, per platform.

    Obsidian writes every vault's absolute path here, so this is one file read
    instead of walking the disk looking for `.obsidian` markers.
    """
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        p = Path(appdata) / "obsidian" / "obsidian.json" if appdata else None
    elif sys.platform == "darwin":
        p = home / "Library" / "Application Support" / "obsidian" / "obsidian.json"
    else:
        p = home / ".config" / "obsidian" / "obsidian.json"
    return p if p and p.is_file() else None


def scan_obsidian(deadline: float | None = None) -> list[Candidate]:
    """Vaults Obsidian itself knows about. A corrupt or absent config is not an
    error — it just means this signal has nothing to say."""
    cfg = obsidian_config_path()
    if cfg is None:
        return []
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        vaults = data.get("vaults") or {}
    except (OSError, ValueError, AttributeError):
        return []
    out: list[Candidate] = []
    for entry in vaults.values():
        raw = (entry or {}).get("path") if isinstance(entry, dict) else None
        if not raw:
            continue
        path = Path(raw)
        if not path.is_dir():
            continue                       # Obsidian keeps deleted vaults listed
        total, exts = count_indexable(path, deadline=deadline)
        if not total:
            continue
        out.append(Candidate(path=path, kind="obsidian", file_count=total,
                             exts=exts, confidence=100,
                             why=f"Obsidian vault - {describe(exts)}"))
    return out


MIN_DENSE_FILES = 5      # fewer than this in one directory is noise
_WALK_SKIP = {"node_modules", "__pycache__", "venv", ".venv", "dist", "build",
              "site-packages", "AppData", "Library"}


def _walk_for_dense(home: Path, deadline: float | None,
                    already: set[Path]) -> tuple[list[Candidate], bool]:
    """Directories holding MIN_DENSE_FILES or more indexable files.

    Only the shallowest match in any chain is kept, so a vault is offered once
    rather than once per subfolder. Returns (candidates, completed).
    """
    out: list[Candidate] = []
    claimed: list[Path] = []
    stack = [home]
    while stack:
        if deadline is not None and time.monotonic() > deadline:
            return out, False
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for d in entries:
            if d.is_dir() and not d.name.startswith(".") \
                    and d.name not in _WALK_SKIP:
                stack.append(d)
        if cur in already or any(cur.is_relative_to(c) for c in claimed):
            continue
        exts: dict[str, int] = {}
        for f in entries:
            if f.is_file() and f.suffix.lower() in DISCOVER_EXTS:
                exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        if sum(exts.values()) >= MIN_DENSE_FILES:
            total, full = count_indexable(cur, deadline=deadline)
            claimed.append(cur)
            out.append(Candidate(path=cur, kind="found", file_count=total,
                                 exts=full, confidence=25, why=describe(full)))
    return out, True


def discover(*, budget_s: float = 1.5, deep: bool = False,
             cwd: Path | None = None,
             home: Path | None = None) -> tuple[list[Candidate], bool]:
    """Folders worth offering, best first, plus whether the search finished.

    Cheapest signals first so the useful answer arrives immediately: Obsidian's
    own vault list, the working directory, then the usual named folders. Only
    the walk of the home directory is budgeted, and only it can be cut short.
    """
    cwd = Path(cwd if cwd is not None else Path.cwd())
    home = Path(home if home is not None else Path.home())

    # ONE deadline for the whole command. Budgeting only the walk left the
    # counting unbounded, and counting a real ~/Documents is what actually took
    # the minutes: `rag discover` ran for over three of them before this.
    deadline = None if deep else time.monotonic() + budget_s

    cands: list[Candidate] = list(scan_obsidian(deadline))
    seen: set[Path] = {c.path for c in cands}

    for c in scan_named([cwd], deadline):
        if c.path not in seen:
            cands.append(Candidate(c.path, "cwd", c.file_count, c.exts, 75,
                                   f"this folder - {c.why}"))
            seen.add(c.path)

    for c in scan_named([home / "Documents", home / "Notes", home / "Desktop"],
                        deadline):
        if c.path not in seen:
            cands.append(c)
            seen.add(c.path)

    walked, complete = _walk_for_dense(home, deadline, seen)
    cands.extend(c for c in walked if c.path not in seen)

    cands.sort(key=lambda c: (-c.confidence, -c.file_count, str(c.path)))
    return cands, complete
