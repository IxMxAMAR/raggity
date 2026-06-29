from raggity.models import Chunk
from raggity.prompts import (format_context, build_user_prompt, parse_cited_ids,
                             verify_citations, chunk_tag, _TAG_PREFIX_LEN)


def _chunk(cid, text):
    return Chunk(text=text, source_path="notes/security.md", title="Security",
                 heading_path="Security", ordinal=0, chunk_id=cid)


def test_chunk_tag_prefix_length():
    """Citation tag uses exactly _TAG_PREFIX_LEN hex chars (16)."""
    # chunk_id longer than prefix — tag should slice to prefix length
    cid = "abcd1234ef567890aabbccdd"
    tag = chunk_tag(1, _chunk(cid, "x"))
    # Extract the hex part between # and ]
    prefix = tag.split("#")[1].rstrip("]")
    assert len(prefix) == _TAG_PREFIX_LEN
    assert prefix == cid[:_TAG_PREFIX_LEN]


def test_chunk_tag_format():
    """Tag format is [doc_N#<hex>] with the full prefix length."""
    cid = "abcd1234ef567890aabbccdd"
    tag = chunk_tag(1, _chunk(cid, "x"))
    expected_prefix = cid[:_TAG_PREFIX_LEN]
    assert tag == f"[doc_1#{expected_prefix}]"


def test_format_context_includes_source_path():
    ctx = format_context([_chunk("abcd1234ef567890aabb", "rotated key")])
    assert "notes/security.md" in ctx and "rotated key" in ctx


def test_build_user_prompt_has_question_and_context():
    cid = "abcd1234ef567890aabb"
    p = build_user_prompt("when did I rotate the key?", [_chunk(cid, "rotated key on 2026-06-01")])
    expected = cid[:_TAG_PREFIX_LEN]
    assert "when did I rotate" in p and f"doc_1#{expected}" in p


def test_parse_cited_ids_16char():
    """parse_cited_ids returns the 16-char hex prefix."""
    cid_prefix = "abcd1234ef567890"
    ids = parse_cited_ids(f"The key was rotated [doc_1#{cid_prefix}] last month.")
    assert ids == [cid_prefix]


def test_parse_cited_ids_old_8char_not_matched():
    """8-char tags no longer match _TAG_RE (regex now requires 16 chars)."""
    ids = parse_cited_ids("Cited [doc_1#abcd1234] here.")
    assert ids == []


def test_verify_marks_supported_and_unsupported():
    # chunk_id must be >= 16 chars so it can be sliced to prefix
    cid = "abcd1234ef567890aabb"
    chunks = [_chunk(cid, "rotated the API key on 2026-06-01")]
    prefix = cid[:_TAG_PREFIX_LEN]
    text = (f"You rotated the API key on 2026-06-01 [doc_1#{prefix}]. "
            f"Also [doc_9#deadbeef01234567] nonsense.")
    cits = verify_citations(text, chunks)
    by_id = {c.chunk_id[:_TAG_PREFIX_LEN]: c for c in cits}
    assert by_id[prefix].supported is True
    assert any(not c.supported for c in cits)


def test_verify_citations_distinct_prefixes():
    """Two chunk_ids that share an 8-hex prefix but differ at 16 → distinct, correct verification."""
    # These share the first 8 chars but differ at position 9+
    cid_a = "abcd1234" + "00000000" + "aabbccdd"   # prefix[:16] = "abcd123400000000"
    cid_b = "abcd1234" + "ffffffff" + "aabbccdd"   # prefix[:16] = "abcd1234ffffffff"
    chunk_a = _chunk(cid_a, "content about alpha systems")
    chunk_b = _chunk(cid_b, "content about beta networks")
    chunks = [chunk_a, chunk_b]
    prefix_a = cid_a[:_TAG_PREFIX_LEN]
    prefix_b = cid_b[:_TAG_PREFIX_LEN]
    # Both should appear in tags distinctly
    tag_a = chunk_tag(1, chunk_a)
    tag_b = chunk_tag(2, chunk_b)
    assert prefix_a in tag_a
    assert prefix_b in tag_b
    assert tag_a != tag_b
    # verify_citations should resolve each separately
    text = (f"Alpha [doc_1#{prefix_a}]. Beta [doc_2#{prefix_b}].")
    cits = verify_citations(text, chunks)
    resolved_ids = {c.chunk_id for c in cits if c.source_path != "?"}
    assert cid_a in resolved_ids
    assert cid_b in resolved_ids
