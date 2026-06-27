from raggity.models import Chunk
from raggity.prompts import (format_context, build_user_prompt, parse_cited_ids,
                             verify_citations, chunk_tag)


def _chunk(cid, text):
    return Chunk(text=text, source_path="notes/security.md", title="Security",
                 heading_path="Security", ordinal=0, chunk_id=cid)


def test_chunk_tag_format():
    tag = chunk_tag(1, _chunk("abcd1234ef", "x"))
    assert tag.startswith("[doc_1#abcd1234]")


def test_format_context_includes_source_path():
    ctx = format_context([_chunk("abcd1234ef", "rotated key")])
    assert "notes/security.md" in ctx and "rotated key" in ctx


def test_build_user_prompt_has_question_and_context():
    p = build_user_prompt("when did I rotate the key?", [_chunk("abcd1234ef", "rotated key on 2026-06-01")])
    assert "when did I rotate" in p and "doc_1#abcd1234" in p


def test_parse_cited_ids():
    ids = parse_cited_ids("The key was rotated [doc_1#abcd1234] last month.")
    assert ids == ["abcd1234"]


def test_verify_marks_supported_and_unsupported():
    chunks = [_chunk("abcd1234ef", "rotated the API key on 2026-06-01")]
    text = "You rotated the API key on 2026-06-01 [doc_1#abcd1234]. Also [doc_9#deadbeef] nonsense."
    cits = verify_citations(text, chunks)
    by_id = {c.chunk_id[:8]: c for c in cits}
    assert by_id["abcd1234"].supported is True
    # unknown id → unsupported citation recorded
    assert any(not c.supported for c in cits)
