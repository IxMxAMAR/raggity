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
