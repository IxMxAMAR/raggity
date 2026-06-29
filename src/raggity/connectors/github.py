"""GitHub repository connector.

Shallow-clones a repository and reads all text files via the readers module.

No extra dependencies beyond the core install — uses stdlib ``subprocess``
and ``tempfile`` for the clone, then the existing ``readers`` dispatch for
file reading.

Usage::

    from raggity.connectors.github import GitHubConnector

    docs = GitHubConnector("https://github.com/owner/repo").fetch()
    # optionally pin to a branch/tag/SHA:
    docs = GitHubConnector("https://github.com/owner/repo", ref="main").fetch()
    # 40-hex commit SHA also works (uses fetch+checkout, not --branch):
    docs = GitHubConnector("https://github.com/owner/repo", ref="a" * 40).fetch()
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from . import Connector
from ..models import Document
from ..readers import SUPPORTED_EXTS, read_file

# Regex for a full 40-character hex commit SHA.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------------
# Clone seam — monkeypatch this in tests (no real git required)
# ---------------------------------------------------------------------------

def _clone(url: str, dest: str) -> None:
    """Shallow-clone *url* (HEAD) into *dest*.

    This is a module-level named seam so tests can monkeypatch it without
    touching subprocess at all::

        monkeypatch.setattr(raggity.connectors.github, "_clone", fake_clone)
    """
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True,
        capture_output=True,
    )


def _clone_ref(url: str, ref: str, dest: str) -> None:
    """Shallow-clone *url* at *ref* into *dest* (branch or tag name).

    NOTE: ``--branch`` only works with branch/tag names, not commit SHAs.
    For SHAs use :func:`_clone_ref_sha`.
    """
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, dest],
        check=True,
        capture_output=True,
    )


def _clone_ref_sha(url: str, sha: str, dest: str) -> None:
    """Fetch a specific commit SHA into *dest* without using ``--branch``.

    ``git clone --branch <sha>`` fails for commit SHAs on most servers.
    Instead we: init an empty repo, add the remote, fetch the exact SHA
    at depth 1, then checkout FETCH_HEAD.
    """
    subprocess.run(["git", "init", dest], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", url],
        check=True, capture_output=True, cwd=dest,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", sha],
        check=True, capture_output=True, cwd=dest,
    )
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        check=True, capture_output=True, cwd=dest,
    )


# ---------------------------------------------------------------------------
# GitHubConnector
# ---------------------------------------------------------------------------

class GitHubConnector(Connector):
    """Fetch all text files from a GitHub (or any git) repository.

    Parameters
    ----------
    repo_url:
        Clone URL, e.g. ``https://github.com/owner/repo``.
    ref:
        Optional branch or tag name to check out.  When given, passes
        ``--branch ref`` to the ``git clone`` call.
    """

    def __init__(self, repo_url: str, ref: str | None = None) -> None:
        self.repo_url = repo_url
        self.ref = ref

    def fetch(self) -> list[Document]:
        """Clone the repository and return one :class:`Document` per text file."""
        docs: list[Document] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            if self.ref:
                if _SHA_RE.fullmatch(self.ref):
                    # 40-hex commit SHA: use fetch+checkout, not --branch
                    _clone_ref_sha(self.repo_url, self.ref, tmpdir)
                else:
                    _clone_ref(self.repo_url, self.ref, tmpdir)
            else:
                _clone(self.repo_url, tmpdir)

            root = Path(tmpdir)
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                relpath = path.relative_to(root).as_posix()
                try:
                    text = read_file(str(path))
                except Exception:  # noqa: BLE001
                    text = None
                if not text:
                    continue
                # Hash raw file bytes — consistent with loader.compute_file_hash.
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                docs.append(
                    Document(
                        path=f"{self.repo_url}#{relpath}",
                        title=path.stem,
                        text=text,
                        file_hash=file_hash,
                        mtime=0.0,
                    )
                )
        return docs
