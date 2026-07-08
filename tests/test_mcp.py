"""Tests for the `rag mcp` MCP server (v0.11.0 T4).

Offline: no real stdio client. Tool registration is checked via FastMCP's
`list_tools`; tool behaviour is checked by invoking the registered tool
functions directly against a seeded temporary Raggity (LLM mocked for `ask`).
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import raggity.llm as llm_mod
from raggity.config import IndexConfig, RaggityConfig, SourcesConfig
from raggity.core import Raggity


class _Block:
    def __init__(self, text): self.text = text


class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


def _seeded_rag(tmp_path) -> Raggity:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# Backups\n\nbackups run nightly to the NAS server")
    (notes / "b.md").write_text("# Sourdough\n\naurora bakery bakes fresh bread each morning")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(notes / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    rag = Raggity(cfg)
    rag.ingest()
    return rag


def _tool_fn(mcp, name):
    """Return the underlying python function registered under *name*."""
    tool = mcp._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool.fn


# --- registration -----------------------------------------------------------

async def test_build_mcp_registers_expected_tools(tmp_path):
    from raggity.mcp_server import build_mcp

    mcp = build_mcp(_seeded_rag(tmp_path))
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"search", "ask", "kb_status"} <= names


async def test_search_tool_schema(tmp_path):
    from raggity.mcp_server import build_mcp

    mcp = build_mcp(_seeded_rag(tmp_path))
    tools = {t.name: t for t in await mcp.list_tools()}
    props = tools["search"].inputSchema["properties"]
    assert "query" in props
    assert "k" in props
    assert "max_context_tokens" in props
    # ask exposes exactly a `question` argument.
    ask_props = tools["ask"].inputSchema["properties"]
    assert "question" in ask_props


# --- search behaviour -------------------------------------------------------

async def test_search_returns_source_blocks(tmp_path):
    from raggity.mcp_server import build_mcp

    mcp = build_mcp(_seeded_rag(tmp_path))
    search = _tool_fn(mcp, "search")
    out = await search("how are backups done?")
    assert "[source:" in out
    assert "NAS" in out


async def test_search_respects_k(tmp_path):
    from raggity.mcp_server import build_mcp

    mcp = build_mcp(_seeded_rag(tmp_path))
    search = _tool_fn(mcp, "search")
    out = await search("bread bakery", k=1)
    # k=1 -> at most one [source:] block.
    assert out.count("[source:") == 1


async def test_search_respects_token_budget(tmp_path):
    from raggity.mcp_server import build_mcp

    mcp = build_mcp(_seeded_rag(tmp_path))
    search = _tool_fn(mcp, "search")
    full = await search("bread bakery backups", k=8)
    tiny = await search("bread bakery backups", k=8, max_context_tokens=10)
    assert len(tiny) < len(full)


async def test_search_empty_index_friendly(tmp_path):
    from raggity.mcp_server import build_mcp

    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str(tmp_path / "none" / "*.md")]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    mcp = build_mcp(Raggity(cfg))
    search = _tool_fn(mcp, "search")
    out = await search("anything")
    assert "no results" in out.lower()


# --- ask behaviour ----------------------------------------------------------

async def test_ask_returns_cited_answer(tmp_path, monkeypatch):
    from raggity.mcp_server import build_mcp

    rag = _seeded_rag(tmp_path)
    # Craft a mock answer that cites a real retrieved chunk (so it verifies as
    # supported and appears in the Sources footer).
    chunks = rag.retriever.retrieve("how are backups done?")
    cid = chunks[0].chunk_id[:16]

    async def _fake_query(prompt, options):
        yield _AssistantMessage(f"Backups run nightly to the NAS [doc_1#{cid}].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    mcp = build_mcp(rag)
    ask = _tool_fn(mcp, "ask")
    out = await ask("how are backups done?")
    assert "NAS" in out
    assert "Sources:" in out
    assert chunks[0].source_path in out


async def test_ask_abstains_passthrough(tmp_path, monkeypatch):
    from raggity.mcp_server import build_mcp
    from raggity.prompts import ABSTAIN_MESSAGE

    rag = _seeded_rag(tmp_path)

    async def _fake_query(prompt, options):
        yield _AssistantMessage(ABSTAIN_MESSAGE)

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    mcp = build_mcp(rag)
    ask = _tool_fn(mcp, "ask")
    out = await ask("what is the meaning of life?")
    assert out == ABSTAIN_MESSAGE
    assert "Sources:" not in out


# --- kb_status behaviour ----------------------------------------------------

async def test_kb_status_counts(tmp_path):
    from raggity.mcp_server import build_mcp

    rag = _seeded_rag(tmp_path)
    mcp = build_mcp(rag)
    status = _tool_fn(mcp, "kb_status")
    out = await status()
    assert "chunks:" in out
    assert "sources: 2" in out
    assert "index_backend: lancedb" in out


async def test_kb_status_does_not_build_heavy_slots(tmp_path):
    from raggity.core import _UNSET
    from raggity.mcp_server import build_mcp

    rag = _seeded_rag(tmp_path)
    # Fresh instance sharing the same on-disk index: heavy slots start unbuilt.
    fresh = Raggity(rag.cfg)
    mcp = build_mcp(fresh)
    status = _tool_fn(mcp, "kb_status")
    await status()
    # kb_status touches only the store; provider/reranker/raw embedder stay unbuilt.
    assert fresh._provider is _UNSET
    assert fresh._reranker is _UNSET
    assert fresh._raw_embedder is _UNSET


# --- lazy import + missing extra -------------------------------------------

def test_import_cli_does_not_import_mcp():
    """`import raggity.cli` must not eagerly import the heavy `mcp` package."""
    code = (
        "import sys\n"
        "import raggity.cli\n"
        "leaked = [m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')]\n"
        "assert not leaked, f'mcp leaked into sys.modules: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_rag_mcp_missing_extra_hint(monkeypatch, capsys):
    """`rag mcp` with the `mcp` extra absent exits 1 and prints the install hint.

    build_mcp raises ImportError(MCP_EXTRA_HINT) when the `mcp` package is missing;
    run_mcp propagates it. Simulate that here without uninstalling anything.
    """
    import typer

    import raggity.mcp_server as mcp_server
    from raggity import cli

    def _boom(config_path=None):
        raise ImportError(mcp_server.MCP_EXTRA_HINT)

    monkeypatch.setattr(mcp_server, "run_mcp", _boom)
    with pytest.raises(typer.Exit) as exc:
        cli.mcp(config=None)
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "raggity[mcp]" in out


# --- pyproject extras -------------------------------------------------------

def test_pyproject_declares_mcp_extra():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert any(dep.startswith("mcp") for dep in extras["mcp"])
    assert any(dep.startswith("mcp") for dep in extras["all"])
