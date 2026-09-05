"""Find document folders worth offering to index.

Metadata only: names, extensions and counts. Nothing here opens a user's
documents, and nothing here indexes anything — it produces candidates that a
human confirms.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .loader import _expand
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


def count_indexable(path: Path) -> tuple[int, dict[str, int]]:
    """How many files under *path* ingest would actually take, by extension.

    Counted through the loader's own glob expansion so the number shown is the
    number ingested — junk directories pruned the same way.
    """
    pattern = str(Path(path) / "**" / "*")
    found = [Path(p) for p in _expand([pattern])]
    exts = Counter(p.suffix.lower() for p in found
                   if p.suffix.lower() in DISCOVER_EXTS and p.is_file())
    return sum(exts.values()), dict(sorted(exts.items()))


def describe(exts: dict[str, int]) -> str:
    """`341 notes` / `1,204 documents - pdf, docx, md`."""
    total = sum(exts.values())
    kinds = ", ".join(e.lstrip(".") for e, _ in
                      sorted(exts.items(), key=lambda kv: -kv[1])[:3])
    noun = "note" if set(exts) <= {".md", ".txt"} else "document"
    return f"{total:,} {noun}{'s' if total != 1 else ''}" + (
        f" - {kinds}" if len(exts) > 1 else "")


def scan_named(roots: list[Path]) -> list[Candidate]:
    """Candidates for a fixed list of directories. Missing or empty ones are
    simply absent — an offer with nothing behind it wastes the user's read."""
    out: list[Candidate] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        total, exts = count_indexable(root)
        if not total:
            continue
        out.append(Candidate(path=root, kind="known", file_count=total,
                             exts=exts, confidence=50, why=describe(exts)))
    return out
