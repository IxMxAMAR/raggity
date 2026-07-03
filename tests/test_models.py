from raggity.models import Document, Chunk, Citation, Answer


def test_document_fields():
    d = Document(path="a.md", title="A", text="hello", file_hash="h", mtime=1.0)
    assert d.path == "a.md" and d.title == "A" and d.text == "hello"


def test_chunk_defaults_score_zero():
    c = Chunk(text="t", source_path="a.md", title="A", heading_path="A > B",
              ordinal=0, chunk_id="id1")
    assert c.score == 0.0


def test_answer_defaults_not_abstained():
    a = Answer(text="ans", citations=[])
    assert a.abstained is False and a.citations == []


def test_citation_supported_flag():
    cit = Citation(chunk_id="id1", source_path="a.md", title="A", supported=True)
    assert cit.supported is True


def test_chunk_vector_defaults_none():
    c = Chunk(text="t", source_path="a.md", title="A", heading_path="A > B",
              ordinal=0, chunk_id="id1")
    assert c.vector is None


def test_chunk_vector_excluded_from_eq():
    c1 = Chunk(text="t", source_path="a.md", title="A", heading_path="A",
               ordinal=0, chunk_id="id1", vector=[1.0, 2.0])
    c2 = Chunk(text="t", source_path="a.md", title="A", heading_path="A",
               ordinal=0, chunk_id="id1", vector=[9.0, 9.0])
    assert c1 == c2


def test_chunk_vector_excluded_from_repr():
    c = Chunk(text="t", source_path="a.md", title="A", heading_path="A",
              ordinal=0, chunk_id="id1", vector=[1.0, 2.0, 3.0])
    assert "vector" not in repr(c)
