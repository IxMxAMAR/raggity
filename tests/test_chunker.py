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


def test_parent_mode_children_share_parent_id_and_text():
    from raggity.chunker import chunk_document
    body = "\n\n".join(f"paragraph {i} " * 30 for i in range(12))
    doc = _doc(f"# A\n\n{body}")
    chunks = chunk_document(doc, parent_document=True,
                            parent_target_tokens=200, child_target_tokens=60)
    assert len(chunks) > 1
    # children of the same parent share parent_id and parent_text
    by_parent = {}
    for c in chunks:
        assert c.parent_id != ""
        assert c.parent_text != ""
        by_parent.setdefault(c.parent_id, set()).add(c.parent_text)
    # each parent_id maps to exactly one parent_text
    assert all(len(texts) == 1 for texts in by_parent.values())
    # at least one parent has multiple children (small-to-big)
    counts = {}
    for c in chunks:
        counts[c.parent_id] = counts.get(c.parent_id, 0) + 1
    assert max(counts.values()) >= 2
    # child text differs from (is shorter than) its parent text
    assert any(len(c.text) < len(c.parent_text) for c in chunks)


def test_hard_split_paragraph_bounds_slices():
    """An oversize paragraph is sliced to <= max_chars, splitting on whitespace."""
    from raggity.chunker import _hard_split_paragraph
    text = "word " * 100000  # ~500k chars, single paragraph, no blank lines
    slices = _hard_split_paragraph(text, 2048)
    assert len(slices) > 100
    assert all(len(s) <= 2048 for s in slices)
    # words are not cut mid-way: reconstructed word stream is identical
    assert "".join(slices).replace(" ", "") == text.replace(" ", "")


def test_oversize_single_paragraph_chunks_are_bounded():
    """A ~1MB single-paragraph doc chunks into many bounded pieces (no colossal
    chunk that would OOM onnxruntime attention)."""
    big = "word " * 200000  # ~1MB, one paragraph, zero blank lines
    chunks = chunk_document(_doc(big, path="big.txt", title="big"),
                            target_tokens=512, overlap_tokens=64)
    assert len(chunks) > 100
    # bounded to ~target_tokens*4 chars, doubled at most by paragraph overlap,
    # plus a small header/margin — nowhere near the whole-document size.
    assert max(len(c.text) for c in chunks) <= 512 * 4 * 2 + 256
    assert max(estimate_tokens(c.text) for c in chunks) <= 512 * 2 + 64


def test_normal_paragraphs_untouched_by_hard_split():
    """Below-target paragraphs are not affected by the oversize split path."""
    from raggity.chunker import _split_body
    body = "\n\n".join(f"short paragraph {i}" for i in range(5))
    pieces = _split_body(body, target_tokens=512, overlap_tokens=64)
    assert pieces == [body]  # one packed piece, byte-identical


def test_default_mode_has_empty_parent_fields():
    from raggity.chunker import chunk_document
    chunks = chunk_document(_doc("# A\n\nshort body"))
    assert chunks[0].parent_id == "" and chunks[0].parent_text == ""
