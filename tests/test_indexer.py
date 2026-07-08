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


def test_batched_ingest_single_upsert(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nalpha content")
    (notes / "b.md").write_text("# B\n\nbeta content")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    calls = {"n": 0}
    real_upsert = store.upsert
    def counting_upsert(chunks, embedder):
        calls["n"] += 1
        return real_upsert(chunks, embedder)
    monkeypatch.setattr(store, "upsert", counting_upsert)
    r = Indexer(emb, store, manifest_path=str(tmp_path / "m.json")).ingest([str(notes / "*.md")])
    assert r.added == 2
    assert calls["n"] == 1          # one batched upsert across both files
    assert store.count() >= 2


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


def test_indexer_calls_ensure_ann(tmp_path, emb, monkeypatch):
    notes = tmp_path / "notes"; notes.mkdir(); (notes / "a.md").write_text("# A\n\nx")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    seen = {}
    monkeypatch.setattr(store, "ensure_ann_index", lambda t: seen.setdefault("t", t))
    Indexer(emb, store, manifest_path=str(tmp_path / "m.json"), ann_threshold=42).ingest([str(notes / "*.md")])
    assert seen["t"] == 42


def test_indexer_issues_one_batched_delete_sources_call(tmp_path, emb, monkeypatch):
    """Changed + vanished paths must go through ONE delete_sources call, not N
    delete_source calls (the batched-delete perf win)."""
    notes = tmp_path / "notes"; notes.mkdir()
    a, b, c = notes / "a.md", notes / "b.md", notes / "c.md"
    _write(a, "# A\n\nalpha")
    _write(b, "# B\n\nbeta")
    _write(c, "# C\n\ngamma")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    indexer.ingest([str(notes / "*.md")])

    _write(a, "# A\n\nalpha v2 changed")  # changed
    b.unlink()                            # vanished

    delete_sources_calls = []
    delete_source_calls = []
    real_delete_sources = store.delete_sources
    real_delete_source = store.delete_source

    def spy_delete_sources(paths):
        delete_sources_calls.append(list(paths))
        return real_delete_sources(paths)

    def spy_delete_source(path):
        delete_source_calls.append(path)
        return real_delete_source(path)

    monkeypatch.setattr(store, "delete_sources", spy_delete_sources)
    monkeypatch.setattr(store, "delete_source", spy_delete_source)

    r = indexer.ingest([str(notes / "*.md")])

    assert delete_source_calls == []            # no per-path calls
    assert len(delete_sources_calls) == 1        # exactly one batched call
    assert set(delete_sources_calls[0]) == {a.as_posix(), b.as_posix()}
    assert r.updated == 1 and r.deleted == 1
    assert all("b.md" not in sp for sp in store.all_source_paths())


# ---------------------------------------------------------------------------
# retrieval.contextual (T10: Anthropic-style contextual retrieval at ingest)
# ---------------------------------------------------------------------------

class _FakeContextProvider:
    def __init__(self, response="Situating context."):
        self.response = response
        self.calls = 0

    async def complete(self, system, prompt):
        self.calls += 1
        return self.response


def test_indexer_contextual_off_is_byte_identical(tmp_path, emb):
    """contextual defaults to False: chunk text in the store is untouched,
    and the provider (if any were passed) is never invoked."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)

    def _boom():
        raise AssertionError("provider must not be built when contextual=False")

    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "manifest.json"),
                      provider=_boom, contextual=False)
    indexer.ingest([str(notes / "*.md")])

    chunks = store.all_chunks()
    assert len(chunks) >= 1
    assert all("Situating context." not in c.text for c in chunks)
    assert all(c.text.startswith("A > A\n\n") for c in chunks)


def test_indexer_contextual_on_prepends_context_to_stored_text(tmp_path, emb):
    notes = tmp_path / "notes"
    notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    provider = _FakeContextProvider("Situating context.")

    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "manifest.json"),
                      provider=provider, contextual=True, ingest_concurrency=4)
    r = indexer.ingest([str(notes / "*.md")])

    assert r.added >= 1
    chunks = store.all_chunks()
    assert len(chunks) >= 1
    assert all(c.text.startswith("Situating context.\n\n") for c in chunks)
    assert provider.calls == len(chunks)


def test_indexer_contextual_only_pays_for_new_or_changed_chunks(tmp_path, emb):
    """A no-op re-ingest (nothing changed) must not call the provider again."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _write(notes / "a.md", "# A\n\nalpha content here")
    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    provider = _FakeContextProvider("ctx")

    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "manifest.json"),
                      provider=provider, contextual=True)
    indexer.ingest([str(notes / "*.md")])
    first_calls = provider.calls
    assert first_calls >= 1

    r2 = indexer.ingest([str(notes / "*.md")])
    assert r2.added == 0 and r2.updated == 0
    assert provider.calls == first_calls  # no new LLM calls on no-op ingest


def test_ingest_report_carries_skipped_needs_extra(tmp_path, emb, monkeypatch):
    """When a reader raises MissingDependencyError, IngestReport records it."""
    import raggity.readers as r
    from raggity.readers import MissingDependencyError

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (notes / "a.md").write_text("# A\n\nalpha content")

    # Simulate missing ocr extra for images
    monkeypatch.setattr(r, "_run_ocr", lambda src: (_ for _ in ()).throw(
        MissingDependencyError(extra="ocr", message="needs ocr")
    ))

    store = LanceDBStore(path=str(tmp_path / "idx"), dim=emb.dim)
    indexer = Indexer(emb, store, manifest_path=str(tmp_path / "m.json"))
    report = indexer.ingest([str(notes / "*.md"), str(notes / "*.png")])

    assert report.added >= 1  # the .md got indexed
    assert report.skipped_needs_extra.get("ocr", 0) == 1
    assert report.skipped_generic == 0
