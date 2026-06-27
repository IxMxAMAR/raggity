import json
from pathlib import Path
import pytest
from raggity.indexer import Indexer
from raggity.store import LanceDBStore
from raggity.embedder import FastEmbedEmbedder


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder()


def _write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")


def test_ingest_adds_then_noop(tmp_path, emb):
    notes = tmp_path / "notes"
    notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    idx_path = str(tmp_path / "idx")
    store = LanceDBStore(path=idx_path, dim=emb.dim)
    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "manifest.json"))

    r1 = indexer.ingest([str(notes / "*.md")])
    assert r1.added >= 1 and store.count() >= 1

    r2 = indexer.ingest([str(notes / "*.md")])
    assert r2.added == 0 and r2.updated == 0 and r2.unchanged >= 1


def test_ingest_detects_change(tmp_path, emb):
    notes = tmp_path / "notes"
    notes.mkdir()
    f = notes / "a.md"
    _write(f, "# A\n\nv1")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    indexer.ingest([str(notes / "*.md")])
    _write(f, "# A\n\nv2 changed content")
    r = indexer.ingest([str(notes / "*.md")])
    assert r.updated >= 1


def test_ingest_deletes_removed_source(tmp_path, emb):
    notes = tmp_path / "notes"
    notes.mkdir()
    a, b = notes / "a.md", notes / "b.md"
    _write(a, "# A\n\nalpha")
    _write(b, "# B\n\nbeta")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    indexer.ingest([str(notes / "*.md")])
    b.unlink()
    r = indexer.ingest([str(notes / "*.md")])
    assert r.deleted >= 1
    assert all("b.md" not in sp for sp in store.all_source_paths())


def test_fingerprint_change_triggers_full_rebuild(tmp_path, emb):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nalpha content")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    Indexer(emb, store, manifest_path=str(tmp_path / "m.json"),
            fingerprint="fp-v1").ingest([str(notes / "*.md")])
    n1 = store.count()
    assert n1 >= 1
    # same files, new fingerprint -> should reset + re-ingest (no duplicate growth)
    r = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"),
                fingerprint="fp-v2").ingest([str(notes / "*.md")])
    assert r.added >= 1           # treated as fresh after reset
    assert store.count() == n1    # not doubled
