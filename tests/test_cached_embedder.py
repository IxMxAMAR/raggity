import json
import threading

from raggity.cached_embedder import CachedEmbedder


class _Inner:
    dim = 3

    def __init__(self):
        self.calls = 0

    def embed_query(self, t):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t)), 0.0, 0.0] for t in texts]


def test_hit_skips_inner(tmp_path):
    inner = _Inner()
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.sqlite"))
    v1 = ce.embed_documents(["alpha", "beta"])
    assert inner.calls == 1
    v2 = ce.embed_documents(["alpha", "beta"])    # all cached
    assert inner.calls == 1                        # inner NOT called again
    assert v1 == v2


def test_partial_miss_only_embeds_new(tmp_path):
    inner = _Inner()
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.sqlite"))
    ce.embed_documents(["alpha"])
    ce.embed_documents(["alpha", "gamma"])         # only gamma is new
    assert inner.calls == 2


def test_dim_and_query_delegate(tmp_path):
    inner = _Inner()
    ce = CachedEmbedder(inner, str(tmp_path / "c.sqlite"))
    assert ce.dim == 3 and ce.embed_query("x") == [1.0, 0.0, 0.0]


def test_corrupt_cache_tolerated(tmp_path):
    p = tmp_path / "embed_cache.sqlite"
    p.write_text("this is not a sqlite database")
    ce = CachedEmbedder(_Inner(), str(p))
    # Degrades to re-embedding (no caching) rather than raising.
    assert ce.embed_documents(["alpha"])[0][0] == 5.0


class _InnerModelA(_Inner):
    """Simulates a model with model_id='model-a' and dim=3."""
    model_id = "model-a"
    dim = 3


class _InnerModelB(_Inner):
    """Simulates a different model with model_id='model-b' and dim=4."""
    model_id = "model-b"
    dim = 4


def test_different_model_ids_produce_different_cache_keys(tmp_path):
    """Same text + different model_id → different cache key → inner called for both."""
    cache_path = str(tmp_path / "embed_cache.sqlite")
    inner_a = _InnerModelA()
    ce_a = CachedEmbedder(inner_a, cache_path)
    # embed with model-a
    ce_a.embed_documents(["hello"])
    assert inner_a.calls == 1

    # Load same cache file with model-b — should NOT get a cache hit
    inner_b = _InnerModelB()
    ce_b = CachedEmbedder(inner_b, cache_path)
    ce_b.embed_documents(["hello"])
    assert inner_b.calls == 1, "model-b must not reuse model-a's cached vector"


def test_same_model_id_still_caches(tmp_path):
    """Same text + same model_id → cache hit on second CachedEmbedder using same file."""
    cache_path = str(tmp_path / "embed_cache.sqlite")
    inner1 = _InnerModelA()
    ce1 = CachedEmbedder(inner1, cache_path)
    ce1.embed_documents(["hello"])
    assert inner1.calls == 1
    ce1.close()

    inner2 = _InnerModelA()
    ce2 = CachedEmbedder(inner2, cache_path)
    ce2.embed_documents(["hello"])
    assert inner2.calls == 0, "same model_id → must be a cache hit"


# --- SQLite-specific behaviour ------------------------------------------

def test_sqlite_roundtrip_fidelity(tmp_path):
    """Vectors survive the float32 BLOB round-trip through the cache."""
    class _Precise(_Inner):
        model_id = "precise"
        dim = 4

        def embed_documents(self, texts):
            self.calls += 1
            return [[0.125, -1.5, 2.75, 100.0] for _ in texts]

    inner = _Precise()
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.sqlite"))
    fresh = ce.embed_documents(["x"])
    cached = ce.embed_documents(["x"])   # hit path returns unpacked BLOB
    assert cached == fresh
    assert cached[0] == [0.125, -1.5, 2.75, 100.0]


def test_json_migration_imports_and_renames(tmp_path):
    """A legacy embed_cache.json alongside is imported once, then renamed .json.bak."""
    import hashlib
    sqlite_path = tmp_path / "embed_cache.sqlite"
    json_path = tmp_path / "embed_cache.json"
    # Pre-seed a legacy JSON cache keyed exactly as CachedEmbedder would key it.
    model_prefix = "model-a|3|"
    key = hashlib.sha256((model_prefix + "hello").encode("utf-8")).hexdigest()
    json_path.write_text(json.dumps({key: [7.0, 8.0, 9.0]}))

    inner = _InnerModelA()
    ce = CachedEmbedder(inner, str(sqlite_path))
    out = ce.embed_documents(["hello"])
    assert inner.calls == 0, "migrated JSON entry must produce a cache hit"
    assert out[0] == [7.0, 8.0, 9.0]
    # Legacy JSON retired to .json.bak
    assert not json_path.exists()
    assert (tmp_path / "embed_cache.json.bak").exists()


def test_threaded_consistency(tmp_path):
    """Concurrent embed_documents from many threads returns correct vectors."""
    inner = _InnerModelA()
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.sqlite"))
    texts = [f"t{i}" for i in range(50)]
    results = {}
    errors = []

    def _work(t):
        try:
            results[t] = ce.embed_documents([t])[0]
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_work, args=(t,)) for t in texts]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    for t in texts:
        assert results[t] == [float(len(t)), 0.0, 0.0]
