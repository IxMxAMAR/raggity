import pytest
from raggity.embedder import FastEmbedEmbedder


@pytest.fixture(scope="module")
def emb():
    return FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", provider="cpu")


def test_dim_is_384(emb):
    assert emb.dim == 384


def test_embed_documents_shape(emb):
    vecs = emb.embed_documents(["hello world", "another doc"])
    assert len(vecs) == 2 and len(vecs[0]) == 384


def test_embed_query_shape(emb):
    v = emb.embed_query("a question")
    assert len(v) == 384


def test_similar_texts_closer_than_dissimilar(emb):
    import numpy as np
    a = np.array(emb.embed_query("how do I back up my files"))
    b = np.array(emb.embed_documents(["backups run nightly to the NAS"])[0])
    c = np.array(emb.embed_documents(["the cat sat on the mat"])[0])
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert cos(a, b) > cos(a, c)
