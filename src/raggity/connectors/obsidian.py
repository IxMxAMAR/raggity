"""Obsidian vault connector.

Walks a local Obsidian vault directory, reads every ``.md`` file, and
normalises ``[[wikilink]]`` / ``[[link|alias]]`` syntax to plain text so
the chunks are useful for embedding without raw bracket noise.

No optional dependencies — pure stdlib + the existing readers module.

Usage::

    from raggity.connectors.obsidian import ObsidianConnector

    docs = ObsidianConnector("/path/to/vault").fetch()
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import Connector
from ..models import Document


# Matches [[Link Text]] and [[Link Text|alias]] — captured group is the
# display text (alias when present, otherwise the link target).
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _normalise_wikilinks(text: str) -> str:
    """Replace ``[[link]]`` / ``[[link|alias]]`` with plain display text."""

    def _replacement(m: re.Match) -> str:
        # Group 2 is the alias (after |); group 1 is the link target.
        return m.group(2) if m.group(2) else m.group(1)

    return _WIKILINK_RE.sub(_replacement, text)


class ObsidianConnector(Connector):
    """Read all Markdown notes from an Obsidian vault directory.

    Wikilinks (``[[Note Name]]`` and ``[[Note Name|display text]]``) are
    normalised to their display text before indexing.

    Parameters
    ----------
    vault_dir:
        Path to the root of the Obsidian vault (the folder that contains
        ``.obsidian/`` and your ``.md`` notes).
    """

    def __init__(self, vault_dir: str) -> None:
        self.vault_dir = Path(vault_dir)

    def fetch(self) -> list[Document]:
        """Return one :class:`Document` per ``.md`` file in the vault."""
        docs: list[Document] = []
        for path in sorted(self.vault_dir.rglob("*.md")):
            if not path.is_file():
                continue
            raw_bytes = path.read_bytes()
            raw = raw_bytes.decode("utf-8", errors="replace")
            text = _normalise_wikilinks(raw)
            # Hash raw file bytes — consistent with loader.compute_file_hash.
            file_hash = hashlib.sha256(raw_bytes).hexdigest()
            docs.append(
                Document(
                    # Use POSIX forward slashes (avoids Windows backslashes).
                    path=Path(path).as_posix(),
                    title=path.stem,
                    text=text,
                    file_hash=file_hash,
                    mtime=path.stat().st_mtime,
                )
            )
        return docs
