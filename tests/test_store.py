import pytest
from raggity.models import Chunk
from raggity.store import LanceDBStore, rrf_fuse
from raggity.embedder import FastEmbedEmbedder


def test_rrf_fuse_rewards_top_ranks():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder()


def _chunk(cid, text, src="a.md", ordinal=0):
    return Chunk(text=text, source_path=src, title="A",
                 heading_path="A", ordinal=ordinal, chunk_id=cid)


def test_upsert_and_count(tmp_path, emb):
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([_chunk("c1", "backups run nightly to the NAS"),
                  _chunk("c2", "the cat sat on the mat", ordinal=1)], emb)
    assert store.count() == 2


def test_vector_search_returns_relevant_first(tmp_path, emb):
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([_chunk("c1", "backups run nightly to the NAS"),
                  _chunk("c2", "the cat sat on the mat", ordinal=1)], emb)
    res = store.vector_search(emb.embed_query("how are backups done"), limit=2)
    assert res[0].chunk_id == "c1"
    assert res[0].score >= res[1].score


def test_text_search_exact_term(tmp_path, emb):
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([_chunk("c1", "rotated the API key on 2026-06-01"),
                  _chunk("c2", "unrelated content", ordinal=1)], emb)
    res = store.text_search("API key", limit=2)
    assert res[0].chunk_id == "c1"


def test_delete_source_and_paths(tmp_path, emb):
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([_chunk("c1", "alpha", src="a.md"),
                  _chunk("c2", "beta", src="b.md", ordinal=1)], emb)
    assert store.all_source_paths() == {"a.md", "b.md"}
    store.delete_source("a.md")
    assert store.all_source_paths() == {"b.md"}
    assert store.count() == 1


def test_upsert_roundtrips_parent_fields(tmp_path, emb):
    from raggity.models import Chunk
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    c = Chunk(text="child text about backups", source_path="a.md", title="A",
              heading_path="A", ordinal=0, chunk_id="c1",
              parent_id="p1", parent_text="the full parent block about backups")
    store.upsert([c], emb)
    res = store.vector_search(emb.embed_query("backups"), limit=1)
    assert res[0].parent_id == "p1"
    assert res[0].parent_text == "the full parent block about backups"


def test_reset_empties_table(tmp_path, emb):
    from raggity.models import Chunk
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([Chunk(text="x", source_path="a.md", title="A", heading_path="A",
                        ordinal=0, chunk_id="c1")], emb)
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_ensure_ann_index_noop_below_threshold(tmp_path, emb):
    from raggity.models import Chunk
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    store.upsert([Chunk(text="x", source_path="a.md", title="A", heading_path="A",
                        ordinal=0, chunk_id="c1")], emb)
    store.ensure_ann_index(threshold=1000)   # below → must NOT raise, no index built
    assert store.count() == 1                # still searchable


def test_from_config_builds_lancedb(tmp_path):
    from raggity.store import LanceDBStore
    from raggity.config import RaggityConfig, IndexConfig
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    s = LanceDBStore.from_config(cfg, 384)
    assert s.count() == 0
