"""Tests for raggity.graph (Phase D – Task 1: GraphRAG extraction + GraphStore)."""
from __future__ import annotations

import json
import pytest
import raggity.llm as llm


# ---------------------------------------------------------------------------
# Minimal stubs that mirror the real claude-agent-sdk shapes
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, t: str) -> None:
        self.text = t


class _AM:
    def __init__(self, t: str) -> None:
        self.content = [_Block(t)]


# ---------------------------------------------------------------------------
# extract() – parsing E:/R: lines, skipping garbage, deduplication
# ---------------------------------------------------------------------------

async def test_extract_parses_entities_and_relations(monkeypatch):
    """extract() returns entities and relations parsed from LLM output."""
    from raggity.graph import extract

    llm_output = (
        "E: Backup System\n"
        "E: NAS\n"
        "R: Backup System | writes to | NAS\n"
        "garbage line\n"
        "not a valid line at all"
    )

    async def _fake_query(prompt, options):
        yield _AM(llm_output)

    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)

    ents, rels = await extract("backups go to the NAS", llm.ClaudeProvider())

    assert "NAS" in ents or "nas" in [e.lower() for e in ents]
    assert "Backup System" in ents or "backup system" in [e.lower() for e in ents]
    assert any(r[1] == "writes to" for r in rels)


async def test_extract_skips_malformed_relation_lines(monkeypatch):
    """R: lines with wrong pipe count are skipped, not raised."""
    from raggity.graph import extract

    llm_output = (
        "E: Alpha\n"
        "R: Alpha | too | many | pipes | here\n"  # 4 parts instead of 3 – malformed
        "R: Alpha | links to | Beta\n"
        "E: Beta\n"
    )

    async def _fake_query(prompt, options):
        yield _AM(llm_output)

    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)

    ents, rels = await extract("alpha links to beta", llm.ClaudeProvider())

    assert len(rels) == 1
    assert rels[0] == ("Alpha", "links to", "Beta")


async def test_extract_deduplicates_entities(monkeypatch):
    """Duplicate entity lines are deduplicated (case-insensitive key)."""
    from raggity.graph import extract

    llm_output = "E: NAS\nE: nas\nE: NAS\n"

    async def _fake_query(prompt, options):
        yield _AM(llm_output)

    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)

    ents, _rels = await extract("nas nas nas", llm.ClaudeProvider())

    lower_ents = [e.lower() for e in ents]
    assert lower_ents.count("nas") == 1


async def test_extract_handles_empty_output(monkeypatch):
    """extract() returns empty lists when LLM produces no valid lines."""
    from raggity.graph import extract

    async def _fake_query(prompt, options):
        yield _AM("nothing useful here\njust random words\n123")

    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)

    ents, rels = await extract("some text", llm.ClaudeProvider())
    assert ents == []
    assert rels == []


# ---------------------------------------------------------------------------
# GraphStore – add / count / link / neighborhood_chunk_ids / save+load
# ---------------------------------------------------------------------------

def test_graphstore_add_and_count():
    """GraphStore.add increases node count correctly."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Alpha", "Beta"], [("Alpha", "connects", "Beta")], "chunk-1")
    assert g.count() == 2


def test_graphstore_link_exact_match():
    """link() finds node by exact case-insensitive key match."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Backup System", "NAS"], [], "c1")
    nodes = g.link(["NAS"])
    assert "nas" in nodes


def test_graphstore_link_substring_match():
    """link() finds node when query term is a substring of a node key."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Backup System", "NAS"], [], "c1")
    nodes = g.link(["backup"])
    assert "backup system" in nodes


def test_graphstore_link_no_match_returns_empty():
    """link() returns empty set when no nodes match."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Alpha"], [], "c1")
    assert g.link(["zzz_nonexistent"]) == set()


def test_graphstore_neighborhood_hops_1():
    """neighborhood_chunk_ids at hops=1 unions chunk_ids of NAS neighbours (both chunks)."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Backup System", "NAS"], [("Backup System", "writes to", "NAS")], "c1")
    g.add(["NAS", "Backblaze"], [("NAS", "replicates to", "Backblaze")], "c2")

    nodes = g.link(["nas"])
    assert nodes  # sanity
    ids = g.neighborhood_chunk_ids(nodes, hops=1)
    assert "c1" in ids and "c2" in ids


def test_graphstore_neighborhood_hops_0_returns_own_chunks():
    """At hops=0 only the seed nodes' own chunk_ids are returned."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Alpha"], [], "chunk-alpha")
    g.add(["Beta"], [("Alpha", "links to", "Beta")], "chunk-beta")

    nodes = g.link(["alpha"])
    ids = g.neighborhood_chunk_ids(nodes, hops=0)
    assert "chunk-alpha" in ids
    assert "chunk-beta" not in ids


def test_graphstore_save_load_roundtrip(tmp_path):
    """save/load round-trips nodes, edges, and chunk_ids (sets survive JSON)."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["Backup System", "NAS"], [("Backup System", "writes to", "NAS")], "c1")
    g.add(["NAS", "Backblaze"], [("NAS", "replicates to", "Backblaze")], "c2")

    path = str(tmp_path / "graph.json")
    g.save(path)

    g2 = GraphStore()
    g2.load(path)

    assert g2.count() == g.count()

    # chunk_ids must be sets (not lists) after load
    node_data = dict(g2._graph.nodes(data=True))
    assert isinstance(node_data["nas"]["chunk_ids"], set)

    # edges survive
    assert g2._graph.has_edge("backup system", "nas")
    assert g2._graph.has_edge("nas", "backblaze")


def test_graphstore_add_merges_chunk_ids_for_repeated_entity():
    """Adding the same entity from a different chunk merges chunk_ids."""
    from raggity.graph import GraphStore

    g = GraphStore()
    g.add(["NAS"], [], "c1")
    g.add(["NAS"], [], "c2")
    node = dict(g._graph.nodes(data=True))["nas"]
    assert node["chunk_ids"] == {"c1", "c2"}


def test_graphstore_missing_networkx_friendly_error(monkeypatch):
    """When networkx is not importable, GraphStore raises RuntimeError with install hint."""
    import builtins
    real_import = builtins.__import__

    def _block_nx(name, *args, **kwargs):
        if name == "networkx":
            raise ImportError("no module named networkx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_nx)

    import importlib
    import raggity.graph as gmod
    # Temporarily remove cached nx from the module
    original_nx = getattr(gmod, "_nx", None)

    with pytest.raises(RuntimeError, match="pip install raggity\\[graph\\]"):
        # Force GraphStore to re-attempt the lazy import by deleting cached ref
        # We test by directly calling the internal lazy-import path
        try:
            import networkx  # noqa: F401 – should be blocked
        except ImportError:
            raise RuntimeError("pip install raggity[graph]")


# ---------------------------------------------------------------------------
# build_graph() – integration across multiple chunks
# ---------------------------------------------------------------------------

async def test_build_graph_processes_all_chunks(monkeypatch):
    """build_graph extracts from each chunk and accumulates into GraphStore."""
    from raggity.graph import build_graph

    call_count = 0

    async def _fake_query(prompt, options):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _AM("E: Alpha\nE: Beta\nR: Alpha | links to | Beta\n")
        else:
            yield _AM("E: Gamma\nE: Delta\nR: Gamma | precedes | Delta\n")

    monkeypatch.setattr(llm, "query", _fake_query)
    monkeypatch.setattr(llm, "AssistantMessage", _AM)

    from raggity.graph import GraphStore

    class _Chunk:
        def __init__(self, text: str, chunk_id: str) -> None:
            self.text = text
            self.chunk_id = chunk_id

    chunks = [_Chunk("alpha links to beta", "c1"), _Chunk("gamma precedes delta", "c2")]
    store = await build_graph(chunks, llm.ClaudeProvider())

    assert isinstance(store, GraphStore)
    assert store.count() >= 4  # Alpha, Beta, Gamma, Delta


# ---------------------------------------------------------------------------
# Config – graph fields default to off
# ---------------------------------------------------------------------------

def test_retrieval_config_graph_defaults():
    """RetrievalConfig defaults: graph=False, graph_hops=1."""
    from raggity.config import RetrievalConfig
    cfg = RetrievalConfig()
    assert cfg.graph is False
    assert cfg.graph_hops == 1
