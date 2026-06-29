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
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.json"))
    v1 = ce.embed_documents(["alpha", "beta"])
    assert inner.calls == 1
    v2 = ce.embed_documents(["alpha", "beta"])    # all cached
    assert inner.calls == 1                        # inner NOT called again
    assert v1 == v2


def test_partial_miss_only_embeds_new(tmp_path):
    inner = _Inner()
    ce = CachedEmbedder(inner, str(tmp_path / "embed_cache.json"))
    ce.embed_documents(["alpha"])
    ce.embed_documents(["alpha", "gamma"])         # only gamma is new
    assert inner.calls == 2


def test_dim_and_query_delegate(tmp_path):
    inner = _Inner()
    ce = CachedEmbedder(inner, str(tmp_path / "c.json"))
    assert ce.dim == 3 and ce.embed_query("x") == [1.0, 0.0, 0.0]


def test_corrupt_cache_tolerated(tmp_path):
    p = tmp_path / "embed_cache.json"
    p.write_text("{ not json")
    ce = CachedEmbedder(_Inner(), str(p))
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
    cache_path = str(tmp_path / "embed_cache.json")
    inner_a = _InnerModelA()
    inner_b = _InnerModelB()
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
    cache_path = str(tmp_path / "embed_cache.json")
    inner1 = _InnerModelA()
    ce1 = CachedEmbedder(inner1, cache_path)
    ce1.embed_documents(["hello"])
    assert inner1.calls == 1

    inner2 = _InnerModelA()
    ce2 = CachedEmbedder(inner2, cache_path)
    ce2.embed_documents(["hello"])
    assert inner2.calls == 0, "same model_id → must be a cache hit"
