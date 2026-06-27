from raggity.config import load_config


def test_rerank_model_override(tmp_path):
    p = tmp_path / "raggity.toml"
    p.write_text('[retrieval]\nrerank_model = "BAAI/bge-reranker-v2-m3"\n')
    assert load_config(str(p)).retrieval.rerank_model == "BAAI/bge-reranker-v2-m3"


def test_embedding_model_override(tmp_path):
    p = tmp_path / "raggity.toml"
    p.write_text('[embedding]\nmodel = "nomic-embed-text-v1.5-Q"\n')
    assert load_config(str(p)).embedding.model == "nomic-embed-text-v1.5-Q"


def test_reranker_constructs_with_custom_model_name():
    # constructor must accept the name without forcing a download at import;
    # FastEmbedReranker downloads on __init__, so only assert the class accepts the kwarg
    import inspect
    from raggity.reranker import FastEmbedReranker
    sig = inspect.signature(FastEmbedReranker.__init__)
    assert "model_name" in sig.parameters


def test_embedder_accepts_custom_model_and_provider():
    import inspect
    from raggity.embedder import FastEmbedEmbedder
    params = inspect.signature(FastEmbedEmbedder.__init__).parameters
    assert "model_name" in params and "provider" in params
