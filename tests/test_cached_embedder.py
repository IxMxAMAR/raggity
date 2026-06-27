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
