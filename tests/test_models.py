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
