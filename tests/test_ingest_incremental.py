"""Task 3 regression tests: mtime fast-path manifest v2 + atomic writes +
connector hash-diff/scope + skip optimize on no-op.

Each of the 8 correctness traps has a dedicated test here.
"""
import json
import os
from pathlib import Path

import pytest

from raggity.embedder import FastEmbedEmbedder
from raggity.indexer import Indexer
from raggity.loader import scan_sources, load_paths
from raggity.store import LanceDBStore


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder()


def _write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")


def _read_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Trap 8 (also 4): loader split + Document.size + scan classification
# ---------------------------------------------------------------------------

def test_loader_fills_document_size_and_scan_classifies(tmp_path):
    notes = tmp_path / "notes"; notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    globs = [str(notes / "*.md")]

    scan = scan_sources(globs, {})
    assert scan.candidates and not scan.unchanged
    assert (notes / "a.md").as_posix() in scan.present

    load = load_paths(scan.candidates, {})
    assert len(load.docs) == 1
    doc = load.docs[0]
    assert doc.size == (notes / "a.md").stat().st_size
    assert doc.size > 0

    # Build a v2 manifest and re-scan: exact (mtime,size) match → unchanged, no candidates
    manifest = {doc.path: {"hash": doc.file_hash, "mtime": doc.mtime, "size": doc.size}}
    scan2 = scan_sources(globs, manifest)
    assert doc.path in scan2.unchanged
    assert scan2.candidates == []


# ---------------------------------------------------------------------------
# Trap 2: mtime lies — touched-but-identical file counts as unchanged,
# is NOT re-chunked/re-embedded, and its manifest mtime is refreshed.
# ---------------------------------------------------------------------------

def test_touched_identical_file_is_unchanged_and_mtime_refreshed(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    f = notes / "a.md"
    _write(f, "# A\n\nalpha content here")
    mpath = str(tmp_path / "m.json")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    idx = Indexer(emb, store, manifest_path=mpath)
    idx.ingest([str(notes / "*.md")])

    old_mtime = _read_manifest(mpath)[f.as_posix()]["mtime"]
    old_hash = _read_manifest(mpath)[f.as_posix()]["hash"]

    # Touch: bump mtime to a distinctly different value, keep content identical.
    new_mtime = old_mtime + 1000.0
    os.utime(f, (new_mtime, new_mtime))

    upsert_calls = {"n": 0}
    embed_calls = {"n": 0}
    real_upsert = store.upsert
    real_embed = emb.embed_documents
    monkeypatch.setattr(store, "upsert",
                        lambda c, e: (upsert_calls.__setitem__("n", upsert_calls["n"] + 1), real_upsert(c, e))[1])
    monkeypatch.setattr(emb, "embed_documents",
                        lambda texts: (embed_calls.__setitem__("n", embed_calls["n"] + 1), real_embed(texts))[1])

    r = idx.ingest([str(notes / "*.md")])

    assert r.unchanged == 1 and r.updated == 0 and r.added == 0
    assert upsert_calls["n"] == 0        # not re-chunked/re-embedded
    assert embed_calls["n"] == 0
    ent = _read_manifest(mpath)[f.as_posix()]
    assert ent["mtime"] == pytest.approx(new_mtime)  # mtime refreshed
    assert ent["hash"] == old_hash                    # hash unchanged (no re-parse needed)


# ---------------------------------------------------------------------------
# Trap 3: v1 migration — flat {path: "hash"} loads via sniff, entries hash
# once (no stat match) then upgrade in place; content NOT re-parsed on match.
# ---------------------------------------------------------------------------

def test_v1_manifest_migrates_and_does_not_reparse(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    f = notes / "a.md"
    _write(f, "# A\n\nalpha content here")
    mpath = str(tmp_path / "m.json")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)

    # Compute the real content hash and seed a v1 (flat string) manifest.
    from raggity.loader import compute_file_hash
    h = compute_file_hash(str(f))
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump({f.as_posix(): h}, fh)

    idx = Indexer(emb, store, manifest_path=mpath)

    # Spy read_file: on a v1 hash match the content must NOT be parsed.
    import raggity.loader as loader_mod
    read_calls = {"n": 0}
    real_read = loader_mod.read_file
    monkeypatch.setattr(loader_mod, "read_file",
                        lambda p: (read_calls.__setitem__("n", read_calls["n"] + 1), real_read(p))[1])

    r = idx.ingest([str(notes / "*.md")])

    assert read_calls["n"] == 0          # hash matched → no parse
    assert r.unchanged == 1 and r.added == 0 and r.updated == 0
    ent = _read_manifest(mpath)[f.as_posix()]
    assert isinstance(ent, dict) and "mtime" in ent and "size" in ent  # upgraded
    assert ent["hash"] == h


# ---------------------------------------------------------------------------
# Trap 1: failed/skipped files NEVER enter the manifest, and re-ingest once
# the reader stops raising (extra installed) picks them up.
# ---------------------------------------------------------------------------

def test_skipped_file_absent_from_manifest_then_ingested_after_extra(tmp_path, emb, monkeypatch):
    import raggity.readers as r
    from raggity.readers import MissingDependencyError

    notes = tmp_path / "notes"; notes.mkdir()
    png = notes / "img.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    mpath = str(tmp_path / "m.json")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    idx = Indexer(emb, store, manifest_path=mpath)

    # Simulate missing 'ocr' extra.
    monkeypatch.setattr(r, "_run_ocr", lambda src: (_ for _ in ()).throw(
        MissingDependencyError(extra="ocr", message="needs ocr")))

    rep = idx.ingest([str(notes / "*.png")])
    assert rep.skipped_needs_extra.get("ocr", 0) == 1
    assert png.as_posix() not in _read_manifest(mpath)   # NOT in manifest

    # Now the extra "appears" — reader stops raising and returns text.
    monkeypatch.setattr(r, "_run_ocr", lambda src: "ocr extracted text now available")
    rep2 = idx.ingest([str(notes / "*.png")])
    assert rep2.added == 1                                # picked up
    assert png.as_posix() in _read_manifest(mpath)


# ---------------------------------------------------------------------------
# Trap 5: no-op ingest skips optimize + ensure_ann_index; delete-only runs them.
# ---------------------------------------------------------------------------

def test_noop_ingest_skips_optimize_and_ann(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    idx = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    idx.ingest([str(notes / "*.md")])

    opt = {"n": 0}; ann = {"n": 0}
    monkeypatch.setattr(store, "optimize", lambda: opt.__setitem__("n", opt["n"] + 1))
    monkeypatch.setattr(store, "ensure_ann_index", lambda t: ann.__setitem__("n", ann["n"] + 1))

    r = idx.ingest([str(notes / "*.md")])   # no-op
    assert r.unchanged == 1
    assert opt["n"] == 0 and ann["n"] == 0


def test_delete_only_ingest_runs_optimize_and_ann(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    a, b = notes / "a.md", notes / "b.md"
    _write(a, "# A\n\nalpha content here")
    _write(b, "# B\n\nbeta content here")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    idx = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    idx.ingest([str(notes / "*.md")])

    b.unlink()   # delete-only change (a unchanged)
    opt = {"n": 0}; ann = {"n": 0}
    real_opt, real_ann = store.optimize, store.ensure_ann_index
    monkeypatch.setattr(store, "optimize", lambda: (opt.__setitem__("n", opt["n"] + 1), real_opt())[1])
    monkeypatch.setattr(store, "ensure_ann_index", lambda t: (ann.__setitem__("n", ann["n"] + 1), real_ann(t))[1])

    r = idx.ingest([str(notes / "*.md")])
    assert r.deleted == 1 and r.unchanged == 1
    assert opt["n"] == 1 and ann["n"] == 1


# ---------------------------------------------------------------------------
# Trap 6: atomic manifest write (.tmp + os.replace).
# ---------------------------------------------------------------------------

def test_manifest_write_is_atomic(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    mpath = str(tmp_path / "m.json")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    idx = Indexer(emb, store, manifest_path=mpath)

    replace_calls = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace",
                        lambda s, d: (replace_calls.append((s, d)), real_replace(s, d))[1])

    idx.ingest([str(notes / "*.md")])

    assert any(s == mpath + ".tmp" and d == mpath for s, d in replace_calls)
    assert not os.path.exists(mpath + ".tmp")   # tmp cleaned up by replace
    assert os.path.isfile(mpath)


# ---------------------------------------------------------------------------
# Trap 7: connector hash-diff skip, scoped deletion isolation, empty-hash
# always upserts, int return preserved, deletion via present/scope.
# ---------------------------------------------------------------------------

def _rag(tmp_path):
    from raggity.config import RaggityConfig, IndexConfig
    from raggity.core import Raggity
    return Raggity(RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx"))))


def test_connector_hash_diff_skips_unchanged(tmp_path, monkeypatch):
    from raggity.models import Document
    rag = _rag(tmp_path)
    doc = Document(path="https://x/a", title="A", text="alpha content here", file_hash="h1", mtime=0.0)

    n1 = rag.ingest_documents([doc])
    assert n1 == 1  # int return preserved

    upserts = {"n": 0}
    real = rag.store.upsert
    monkeypatch.setattr(rag.store, "upsert",
                        lambda c, e: (upserts.__setitem__("n", upserts["n"] + 1), real(c, e))[1])

    n2 = rag.ingest_documents([doc])   # identical hash → skip
    assert n2 == 1
    assert upserts["n"] == 0            # no re-chunk/re-embed


def test_connector_empty_hash_always_upserts(tmp_path, monkeypatch):
    from raggity.models import Document
    rag = _rag(tmp_path)
    doc = Document(path="/fake/a.md", title="A", text="server content here", file_hash="", mtime=0.0)
    rag.ingest_documents([doc])

    upserts = {"n": 0}
    real = rag.store.upsert
    monkeypatch.setattr(rag.store, "upsert",
                        lambda c, e: (upserts.__setitem__("n", upserts["n"] + 1), real(c, e))[1])
    rag.ingest_documents([doc])        # empty hash never matches
    assert upserts["n"] == 1


def test_connector_scoped_deletion_isolation(tmp_path):
    from raggity.models import Document
    rag = _rag(tmp_path)
    a1 = Document(path="scopeA/one", title="1", text="a-one content", file_hash="a1", mtime=0.0)
    a2 = Document(path="scopeA/two", title="2", text="a-two content", file_hash="a2", mtime=0.0)
    b1 = Document(path="scopeB/one", title="1", text="b-one content", file_hash="b1", mtime=0.0)

    rag.ingest_documents([a1, a2], scope="scopeA/")
    rag.ingest_documents([b1], scope="scopeB/")

    sources = rag.store.all_source_paths()
    assert {"scopeA/one", "scopeA/two", "scopeB/one"} <= sources

    # Re-ingest scope A without a2 → a2 pruned, scope B untouched.
    rag.ingest_documents([a1], scope="scopeA/")
    sources = rag.store.all_source_paths()
    assert "scopeA/two" not in sources
    assert "scopeA/one" in sources
    assert "scopeB/one" in sources     # other scope intact

    manifest = rag._load_connector_manifest()
    assert "scopeA/two" not in manifest and "scopeB/one" in manifest
