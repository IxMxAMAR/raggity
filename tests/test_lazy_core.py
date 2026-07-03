"""Task 4: lazy component construction + shared tenant models + deferred tables."""
import threading

from raggity.config import RaggityConfig, SourcesConfig, IndexConfig
from raggity.core import Raggity, _UNSET
from raggity.models import Chunk


def _md(tmp_path, text="# A\n\nbackups run nightly to the NAS"):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text(text)
    return [str(notes / "*.md")]


# --- deferred table + light status --------------------------------------

def test_fresh_index_status_no_table_no_heavy_build(tmp_path):
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    rag = Raggity(cfg)
    st = rag.status()
    assert st["chunks"] == 0
    assert rag.store._tbl is None, "fresh index must not create a table on read"
    assert rag._raw_embedder is _UNSET, "status must not build the embedder"


def test_first_write_creates_table(tmp_path):
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    rag = Raggity(cfg)
    assert rag.store._tbl is None
    rag.store.upsert([Chunk("hi there", "a.md", "A", "A", 0, "c1")], rag.embedder)
    assert rag.store._tbl is not None
    assert rag.status()["chunks"] == 1


def test_status_on_seeded_index_builds_nothing_heavy(tmp_path):
    cfg = RaggityConfig(sources=SourcesConfig(include=_md(tmp_path)),
                        index=IndexConfig(path=str(tmp_path / "idx")))
    Raggity(cfg).ingest()          # seed with a separate instance
    fresh = Raggity(cfg)
    st = fresh.status()
    assert st["chunks"] >= 1
    assert fresh._raw_embedder is _UNSET, "status on a seeded index must not build the embedder"


# --- shared tenant models ------------------------------------------------

def test_tenant_shares_raw_embedder_reranker_provider(tmp_path):
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "base")))
    cfg.embedding.cache = True
    base = Raggity(cfg)
    tenant = base.for_namespace("alice")

    assert tenant.raw_embedder is base.raw_embedder
    assert tenant.provider is base.provider
    assert tenant.reranker is base.reranker      # rerank=True by default

    # Distinct CachedEmbedder wrappers with distinct cache paths, shared raw model.
    assert tenant.embedder is not base.embedder
    assert tenant.embedder._inner is base.embedder._inner
    assert tenant.embedder._path != base.embedder._path


async def test_tenant_close_preserves_shared_provider(tmp_path, monkeypatch):
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "base")))
    base = Raggity(cfg)
    prov = base.provider
    calls = []

    async def _spy_aclose():
        calls.append(1)

    monkeypatch.setattr(prov, "aclose", _spy_aclose)

    tenant = base.for_namespace("alice")
    assert tenant.provider is prov
    await tenant.close()
    assert calls == [], "tenant must not close the shared provider"

    await base.close()
    assert calls == [1], "root must close its own provider exactly once"


# --- concurrency ---------------------------------------------------------

def test_concurrent_cold_store_builds_once(tmp_path, monkeypatch):
    import raggity.store as store_mod
    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    rag = Raggity(cfg)

    real_from_config = store_mod.LanceDBStore.from_config  # bound classmethod
    calls = []

    def _spy(cfg_, dim):
        calls.append(1)
        return real_from_config(cfg_, dim)

    monkeypatch.setattr(store_mod.LanceDBStore, "from_config", _spy)

    n = 8
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def _acc():
        barrier.wait()
        s = rag.store
        with lock:
            results.append(s)

    threads = [threading.Thread(target=_acc) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, "store must be built exactly once under concurrent cold access"
    assert all(r is results[0] for r in results)


# --- store from_config int + callable ------------------------------------

def test_from_config_callable_not_invoked_when_table_exists(tmp_path):
    from raggity.store import LanceDBStore
    from raggity.embedder import FastEmbedEmbedder

    cfg = RaggityConfig(index=IndexConfig(path=str(tmp_path / "idx")))
    emb = FastEmbedEmbedder()

    # int dim keeps working; creates the table on write.
    s1 = LanceDBStore.from_config(cfg, 384)
    s1.upsert([Chunk("hello world", "a.md", "A", "A", 0, "c1")], emb)
    assert s1.count() == 1

    # callable dim: opening an existing table must NOT invoke the callable.
    called = []

    def _dim():
        called.append(1)
        return emb.dim

    s2 = LanceDBStore.from_config(cfg, _dim)
    assert s2.count() == 1
    assert called == [], "callable dim must not be invoked when the table already exists"


def test_noop_ingest_builds_neither_embedder_nor_store(tmp_path):
    """A second (no-op) ingest must not load the embedding model or open the store.

    The mtime fast-path classifies everything unchanged, the stored fingerprint
    matches on model+params (dim borrowed from the stored string), and the
    Indexer resolves its embedder/store callables only on actual work.
    """
    from raggity.config import IndexConfig, RaggityConfig, SourcesConfig

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nalpha content here", encoding="utf-8")
    cfg = RaggityConfig(
        sources=SourcesConfig(include=[str((notes / "*.md").as_posix())]),
        index=IndexConfig(path=str(tmp_path / "idx")),
    )
    first = Raggity(cfg)
    report = first.ingest()
    assert report.added == 1

    fresh = Raggity(cfg)
    report2 = fresh.ingest()
    assert report2.unchanged == 1 and report2.added == 0
    assert fresh._raw_embedder is _UNSET, "no-op ingest must not build the embedder"
    assert fresh._store is _UNSET, "no-op ingest must not open the store"
