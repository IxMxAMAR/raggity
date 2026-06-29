"""Tests for the Connector framework and WebConnector.

Also covers:
- CLI ``rag ingest-url``
- Config ``sources.urls`` ingested by ``Raggity.ingest()``
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# WebConnector — mocked seams (no real HTTP)
# ---------------------------------------------------------------------------

def test_web_connector_yields_document(monkeypatch):
    import raggity.connectors.web as w
    monkeypatch.setattr(w, "_fetch", lambda url: "<html><h1>Hi</h1><p>page body</p></html>")
    monkeypatch.setattr(w, "_extract", lambda html, url: ("Hi", "page body about backups"))
    docs = w.WebConnector("https://example.com/x").fetch()
    assert docs and "page body" in docs[0].text and docs[0].path == "https://example.com/x"


def test_connector_registry_resolves():
    from raggity.registry import resolve
    assert resolve("connector", "web").__name__ == "WebConnector"


def test_web_connector_document_fields(monkeypatch):
    """Document must have correct path, title, text, file_hash, and mtime."""
    import hashlib
    import raggity.connectors.web as w
    monkeypatch.setattr(w, "_fetch", lambda url: "<html>...</html>")
    monkeypatch.setattr(w, "_extract", lambda html, url: ("My Title", "Some content here"))
    docs = w.WebConnector("https://example.com/doc").fetch()
    assert len(docs) == 1
    d = docs[0]
    assert d.path == "https://example.com/doc"
    assert d.title == "My Title"
    assert d.text == "Some content here"
    assert d.file_hash == hashlib.sha256("Some content here".encode()).hexdigest()
    assert d.mtime == 0.0


def test_web_connector_depth0_no_crawl(monkeypatch):
    """depth=0 means only the start URL, no BFS."""
    import raggity.connectors.web as w

    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        return "<html><a href='https://example.com/other'>link</a><p>body</p></html>"

    monkeypatch.setattr(w, "_fetch", fake_fetch)
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", "body"))

    docs = w.WebConnector("https://example.com/start", depth=0).fetch()
    assert len(docs) == 1
    assert len(fetch_calls) == 1


def test_web_connector_depth1_crawls_same_domain(monkeypatch):
    """depth=1 should BFS one level into same-domain links."""
    import raggity.connectors.web as w

    pages = {
        "https://example.com/start": (
            "<html><a href='https://example.com/page2'>p2</a>"
            "<a href='https://other.com/ext'>ext</a><p>start body</p></html>"
        ),
        "https://example.com/page2": "<html><p>page2 body</p></html>",
    }

    monkeypatch.setattr(w, "_fetch", lambda url: pages.get(url, "<html><p>empty</p></html>"))
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", f"body of {url}"))

    docs = w.WebConnector("https://example.com/start", depth=1, same_domain=True).fetch()
    paths = {d.path for d in docs}
    assert "https://example.com/start" in paths
    assert "https://example.com/page2" in paths
    assert "https://other.com/ext" not in paths


def test_web_connector_deduplicates(monkeypatch):
    """Duplicate links in BFS are visited only once."""
    import raggity.connectors.web as w

    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        # Both pages link to each other
        return (
            "<html>"
            "<a href='https://example.com/a'>a</a>"
            "<a href='https://example.com/b'>b</a>"
            "<p>content</p></html>"
        )

    monkeypatch.setattr(w, "_fetch", fake_fetch)
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", "content"))

    docs = w.WebConnector("https://example.com/a", depth=1).fetch()
    # Should only visit /a and /b once each
    assert len(set(fetch_calls)) == len(fetch_calls)


def test_connector_abc_cannot_instantiate():
    """The Connector ABC must not be directly instantiable."""
    from raggity.connectors import Connector
    with pytest.raises(TypeError):
        Connector()  # type: ignore


def test_web_connector_is_connector_subclass():
    from raggity.connectors import Connector
    from raggity.connectors.web import WebConnector
    assert issubclass(WebConnector, Connector)


def test_fetch_missing_dep_raises(monkeypatch):
    """_fetch raises RuntimeError when httpx is not available."""
    import builtins
    import raggity.connectors.web as w

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(RuntimeError, match="httpx"):
        w._fetch("https://example.com")


def test_extract_missing_dep_raises(monkeypatch):
    """_extract raises RuntimeError when trafilatura is not available."""
    import builtins
    import raggity.connectors.web as w

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "trafilatura":
            raise ImportError("no trafilatura")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(RuntimeError, match="trafilatura"):
        w._extract("<html><p>hi</p></html>", "https://example.com")


# ---------------------------------------------------------------------------
# CLI ingest-url
# ---------------------------------------------------------------------------

def test_cli_ingest_url(tmp_path, monkeypatch):
    """rag ingest-url <url> adds docs without deleting existing sources."""
    import raggity.connectors.web as w
    import raggity.cli as cli_mod
    from typer.testing import CliRunner

    monkeypatch.setattr(w, "_fetch", lambda url: "<html><p>web content</p></html>")
    monkeypatch.setattr(w, "_extract", lambda html, url: ("Web Page", "web content here"))

    cfg = tmp_path / "raggity.toml"
    cfg.write_text(
        f'[sources]\ninclude = []\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, [
        "ingest-url", "https://example.com/page", "--config", str(cfg)
    ])
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output


# ---------------------------------------------------------------------------
# Config sources.urls
# ---------------------------------------------------------------------------

def test_config_sources_urls_default_empty():
    """sources.urls defaults to empty list."""
    from raggity.config import RaggityConfig
    cfg = RaggityConfig()
    assert cfg.sources.urls == []


def test_config_sources_urls_parsed(tmp_path):
    """sources.urls is parsed from raggity.toml correctly."""
    from raggity.config import load_config
    cfg_file = tmp_path / "raggity.toml"
    cfg_file.write_text('[sources]\nurls = ["https://example.com", "https://docs.example.com"]\n')
    cfg = load_config(str(cfg_file))
    assert cfg.sources.urls == ["https://example.com", "https://docs.example.com"]


# ---------------------------------------------------------------------------
# ObsidianConnector
# ---------------------------------------------------------------------------

def test_obsidian_reads_vault_with_wikilinks(tmp_path):
    from raggity.connectors.obsidian import ObsidianConnector
    (tmp_path / "note.md").write_text("# Note\n\nSee [[Other Note]] for backups.")
    docs = ObsidianConnector(str(tmp_path)).fetch()
    assert docs and "backups" in docs[0].text


def test_obsidian_normalises_wikilink_to_link_text(tmp_path):
    """[[Target]] → 'Target'; [[Target|Alias]] → 'Alias'."""
    from raggity.connectors.obsidian import ObsidianConnector
    (tmp_path / "note.md").write_text("See [[Other Note]] and [[Target|Alias Display]] here.")
    docs = ObsidianConnector(str(tmp_path)).fetch()
    assert docs
    text = docs[0].text
    assert "[[" not in text
    assert "Other Note" in text
    assert "Alias Display" in text
    assert "Target" not in text or "Alias Display" in text  # alias replaces target


def test_obsidian_multiple_notes(tmp_path):
    """Each .md file becomes one Document."""
    from raggity.connectors.obsidian import ObsidianConnector
    (tmp_path / "a.md").write_text("# A\nfirst note")
    (tmp_path / "b.md").write_text("# B\nsecond note")
    docs = ObsidianConnector(str(tmp_path)).fetch()
    assert len(docs) == 2


def test_obsidian_document_path_is_absolute(tmp_path):
    from raggity.connectors.obsidian import ObsidianConnector
    (tmp_path / "note.md").write_text("content")
    docs = ObsidianConnector(str(tmp_path)).fetch()
    # Path is POSIX (forward slashes); compare using as_posix() of tmp_path
    assert docs and tmp_path.as_posix() in docs[0].path


def test_obsidian_path_uses_forward_slashes(tmp_path):
    """Document path must use POSIX forward slashes, not Windows backslashes."""
    from raggity.connectors.obsidian import ObsidianConnector
    (tmp_path / "note.md").write_text("# Note\n\nSome content.")
    docs = ObsidianConnector(str(tmp_path)).fetch()
    assert docs
    assert "\\" not in docs[0].path


def test_obsidian_hash_matches_raw_bytes(tmp_path):
    """file_hash must equal sha256 of raw file bytes (matching loader.compute_file_hash)."""
    import hashlib
    from raggity.connectors.obsidian import ObsidianConnector
    raw_bytes = "# Note\n\nContent with [[wikilink]] here.".encode("utf-8")
    note_path = tmp_path / "note.md"
    note_path.write_bytes(raw_bytes)
    docs = ObsidianConnector(str(tmp_path)).fetch()
    assert docs
    expected = hashlib.sha256(raw_bytes).hexdigest()
    assert docs[0].file_hash == expected


def test_obsidian_registry_resolves():
    from raggity.registry import resolve
    assert resolve("connector", "obsidian").__name__ == "ObsidianConnector"


# ---------------------------------------------------------------------------
# GitHubConnector
# ---------------------------------------------------------------------------

def test_github_connector_clones_and_reads(tmp_path, monkeypatch):
    import raggity.connectors.github as g
    def fake_clone(url, dest):
        from pathlib import Path
        (Path(dest) / "README.md").write_text("# Repo\n\nproject docs here")
    monkeypatch.setattr(g, "_clone", fake_clone)
    docs = g.GitHubConnector("https://github.com/x/y").fetch()
    assert any("project docs" in d.text for d in docs)


def test_github_connector_doc_path_contains_url(tmp_path, monkeypatch):
    import raggity.connectors.github as g
    def fake_clone(url, dest):
        from pathlib import Path
        (Path(dest) / "README.md").write_text("content")
    monkeypatch.setattr(g, "_clone", fake_clone)
    docs = g.GitHubConnector("https://github.com/x/y").fetch()
    assert docs and docs[0].path.startswith("https://github.com/x/y#")


def test_github_connector_skips_unsupported_files(tmp_path, monkeypatch):
    """Binary/unknown extension files are ignored."""
    import raggity.connectors.github as g
    def fake_clone(url, dest):
        from pathlib import Path
        (Path(dest) / "README.md").write_text("text file")
        (Path(dest) / "binary.bin").write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(g, "_clone", fake_clone)
    docs = g.GitHubConnector("https://github.com/x/y").fetch()
    paths = [d.path for d in docs]
    assert any("README.md" in p for p in paths)
    assert not any(".bin" in p for p in paths)


def test_github_sha_ref_uses_fetch_checkout_not_branch_flag(monkeypatch):
    """A 40-hex SHA ref must use _clone_ref_sha (fetch+checkout), NOT _clone_ref (--branch)."""
    import raggity.connectors.github as g

    calls = []

    def fake_clone_ref(url, ref, dest):
        calls.append(("clone_ref", ref))

    def fake_clone_sha(url, sha, dest):
        calls.append(("clone_sha", sha))
        from pathlib import Path
        (Path(dest) / "README.md").write_text("sha content")

    monkeypatch.setattr(g, "_clone_ref", fake_clone_ref)
    monkeypatch.setattr(g, "_clone_ref_sha", fake_clone_sha)

    sha = "a" * 40
    g.GitHubConnector("https://github.com/x/y", ref=sha).fetch()
    assert any(c[0] == "clone_sha" for c in calls), f"Expected clone_sha, got: {calls}"
    assert all(c[0] != "clone_ref" for c in calls), f"clone_ref called for SHA: {calls}"


def test_github_branch_ref_uses_clone_ref_not_sha_path(monkeypatch):
    """A branch/tag ref (not a 40-hex SHA) must use _clone_ref, not _clone_ref_sha."""
    import raggity.connectors.github as g

    calls = []

    def fake_clone_ref(url, ref, dest):
        calls.append(("clone_ref", ref))
        from pathlib import Path
        (Path(dest) / "README.md").write_text("branch content")

    def fake_clone_sha(url, sha, dest):
        calls.append(("clone_sha", sha))

    monkeypatch.setattr(g, "_clone_ref", fake_clone_ref)
    monkeypatch.setattr(g, "_clone_ref_sha", fake_clone_sha)

    g.GitHubConnector("https://github.com/x/y", ref="main").fetch()
    assert any(c[0] == "clone_ref" for c in calls)
    assert all(c[0] != "clone_sha" for c in calls)


def test_github_file_hash_uses_raw_bytes(tmp_path, monkeypatch):
    """file_hash must equal sha256 of raw file bytes (matching loader.compute_file_hash).

    We verify by patching the clone to write known bytes, then comparing
    the Document's file_hash against sha256(raw_bytes).  This is the same
    convention as loader.compute_file_hash which opens in 'rb' mode.
    """
    import hashlib
    import raggity.connectors.github as g

    # Pure UTF-8 content — hash(raw_bytes) is the expected convention.
    raw_bytes = b"# Repo\n\nproject docs here"

    def fake_clone(url, dest):
        from pathlib import Path
        (Path(dest) / "README.md").write_bytes(raw_bytes)

    monkeypatch.setattr(g, "_clone", fake_clone)
    docs = g.GitHubConnector("https://github.com/x/y").fetch()
    assert docs
    # Convention: hash the raw file bytes, NOT the decoded-then-re-encoded text.
    expected = hashlib.sha256(raw_bytes).hexdigest()
    assert docs[0].file_hash == expected


def test_github_registry_resolves():
    from raggity.registry import resolve
    assert resolve("connector", "github").__name__ == "GitHubConnector"


# ---------------------------------------------------------------------------
# CLI ingest-repo and ingest-obsidian
# ---------------------------------------------------------------------------

def test_cli_ingest_repo(tmp_path, monkeypatch):
    """rag ingest-repo <url> clones and indexes the repo."""
    import raggity.connectors.github as g
    import raggity.cli as cli_mod
    from typer.testing import CliRunner

    def fake_clone(url, dest):
        from pathlib import Path
        (Path(dest) / "README.md").write_text("repo readme content")

    monkeypatch.setattr(g, "_clone", fake_clone)

    cfg = tmp_path / "raggity.toml"
    cfg.write_text(
        f'[sources]\ninclude = []\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, [
        "ingest-repo", "https://github.com/x/y", "--config", str(cfg)
    ])
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output


def test_cli_ingest_obsidian(tmp_path):
    """rag ingest-obsidian <vault> reads notes and indexes them."""
    import raggity.cli as cli_mod
    from typer.testing import CliRunner

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n\nSome content here.")

    cfg = tmp_path / "raggity.toml"
    cfg.write_text(
        f'[sources]\ninclude = []\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, [
        "ingest-obsidian", str(vault), "--config", str(cfg)
    ])
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output


# ---------------------------------------------------------------------------

def test_web_connector_max_pages_limits_crawl(monkeypatch):
    """BFS crawl must stop enqueuing once max_pages visited URLs is reached."""
    import raggity.connectors.web as w

    # Create a link graph: /start → /p1 → /p2 → /p3 → /p4 → /p5 (6 pages total)
    def make_page(next_url=None):
        link = f"<a href='{next_url}'>next</a>" if next_url else ""
        return f"<html><p>body</p>{link}</html>"

    pages = {
        "https://example.com/start": make_page("https://example.com/p1"),
        "https://example.com/p1": make_page("https://example.com/p2"),
        "https://example.com/p2": make_page("https://example.com/p3"),
        "https://example.com/p3": make_page("https://example.com/p4"),
        "https://example.com/p4": make_page("https://example.com/p5"),
        "https://example.com/p5": make_page(),
    }
    monkeypatch.setattr(w, "_fetch", lambda url: pages.get(url, "<html><p>empty</p></html>"))
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", f"body {url}"))

    # With max_pages=3 and depth=10, crawl should stop after 3 pages
    docs = w.WebConnector("https://example.com/start", depth=10, max_pages=3).fetch()
    assert len(docs) == 3


def test_web_connector_max_pages_default_allows_normal_crawl(monkeypatch):
    """Default max_pages must not block small crawls."""
    import raggity.connectors.web as w

    pages = {
        "https://example.com/a": "<html><a href='https://example.com/b'>b</a><p>body a</p></html>",
        "https://example.com/b": "<html><p>body b</p></html>",
    }
    monkeypatch.setattr(w, "_fetch", lambda url: pages.get(url, "<html><p>empty</p></html>"))
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", f"body {url}"))

    docs = w.WebConnector("https://example.com/a", depth=1).fetch()
    assert len(docs) == 2


def test_web_connector_depth0_ignores_max_pages(monkeypatch):
    """depth=0 single-fetch behavior must be unchanged regardless of max_pages."""
    import raggity.connectors.web as w

    monkeypatch.setattr(w, "_fetch", lambda url: "<html><a href='https://example.com/b'>b</a><p>body</p></html>")
    monkeypatch.setattr(w, "_extract", lambda html, url: ("T", "body"))

    docs = w.WebConnector("https://example.com/a", depth=0, max_pages=1).fetch()
    assert len(docs) == 1


def test_ingest_calls_web_connector_for_urls(tmp_path, monkeypatch):
    """Raggity.ingest() fetches each sources.urls entry via WebConnector."""
    import raggity.connectors.web as w

    fetched_urls = []

    def fake_fetch(url):
        fetched_urls.append(url)
        return "<html><p>content</p></html>"

    monkeypatch.setattr(w, "_fetch", fake_fetch)
    monkeypatch.setattr(w, "_extract", lambda html, url: ("Title", "content"))

    from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
    cfg = RaggityConfig(
        sources=SourcesConfig(urls=["https://example.com/a", "https://example.com/b"]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    from raggity.core import Raggity
    rag = Raggity(cfg)
    report = rag.ingest()
    assert "https://example.com/a" in fetched_urls
    assert "https://example.com/b" in fetched_urls
    assert report.added == 2
