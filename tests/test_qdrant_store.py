import pytest
from raggity.models import Chunk
from raggity.qdrant_store import QdrantStore
from raggity.embedder import FastEmbedEmbedder


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder()


def _store(dim):
    return QdrantStore(location=":memory:", dim=dim, collection="t")


def _chunk(cid, text, src="a.md", ordinal=0, parent_id="", parent_text=""):
    return Chunk(text=text, source_path=src, title="A", heading_path="A",
                 ordinal=ordinal, chunk_id=cid, parent_id=parent_id, parent_text=parent_text)


def test_upsert_count_and_vector_search(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "backups run nightly to the NAS"),
              _chunk("c2", "the cat sat on the mat", ordinal=1)], emb)
    assert s.count() == 2
    res = s.vector_search(emb.embed_query("how are backups done"), limit=2)
    assert res[0].chunk_id == "c1" and res[0].score >= res[1].score


def test_text_search_keyword(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "rotated the API key on 2026"),
              _chunk("c2", "unrelated content", ordinal=1)], emb)
    res = s.text_search("API key", limit=5)
    assert any(c.chunk_id == "c1" for c in res)


def test_delete_source_and_paths(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "alpha", src="a.md"),
              _chunk("c2", "beta", src="b.md", ordinal=1)], emb)
    assert s.all_source_paths() == {"a.md", "b.md"}
    s.delete_source("a.md")
    assert s.all_source_paths() == {"b.md"} and s.count() == 1


def test_parent_fields_roundtrip(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "child", parent_id="p1", parent_text="PARENT FULL")], emb)
    res = s.vector_search(emb.embed_query("child"), limit=1)
    assert res[0].parent_id == "p1" and res[0].parent_text == "PARENT FULL"


def test_reset_empties(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "x")], emb)
    assert s.count() == 1
    s.reset()
    assert s.count() == 0


def test_ensure_ann_index_noop(emb):
    _store(emb.dim).ensure_ann_index(1)   # no-op, must not raise


def test_get_by_chunk_ids(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "alpha"), _chunk("c2", "beta", ordinal=1)], emb)
    got = s.get_by_chunk_ids(["c2"])
    assert len(got) == 1 and got[0].chunk_id == "c2"


def test_get_by_chunk_ids_multiple(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "alpha"), _chunk("c2", "beta", ordinal=1),
              _chunk("c3", "gamma", ordinal=2)], emb)
    got = s.get_by_chunk_ids(["c1", "c3"])
    ids = {c.chunk_id for c in got}
    assert ids == {"c1", "c3"}


def test_get_by_chunk_ids_empty(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "alpha")], emb)
    assert s.get_by_chunk_ids([]) == []


def test_get_by_chunk_ids_missing(emb):
    s = _store(emb.dim)
    s.upsert([_chunk("c1", "alpha")], emb)
    assert s.get_by_chunk_ids(["nonexistent"]) == []
