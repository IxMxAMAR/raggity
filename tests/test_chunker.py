from raggity.models import Document
from raggity.chunker import chunk_document, estimate_tokens


def _doc(text, path="a.md", title="A"):
    return Document(path=path, title=title, text=text, file_hash="h", mtime=1.0)


def test_short_doc_is_one_chunk():
    chunks = chunk_document(_doc("# A\n\nshort body"))
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0


def test_contextual_header_prefixed():
    chunks = chunk_document(_doc("# A\n\n## Sub\n\nbody text here"))
    assert chunks[0].text.startswith("A")
    assert "Sub" in chunks[0].text  # heading path embedded in header


def test_heading_path_tracked():
    chunks = chunk_document(_doc("# A\n\n## Sub\n\nbody"))
    assert "Sub" in chunks[0].heading_path


def test_long_doc_splits_with_unique_ids():
    body = "\n\n".join(f"paragraph number {i} " * 40 for i in range(20))
    chunks = chunk_document(_doc(f"# A\n\n{body}"), target_tokens=128, overlap_tokens=16)
    assert len(chunks) > 1
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_estimate_tokens_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
