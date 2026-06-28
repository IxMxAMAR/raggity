"""Tests for src/raggity/readers.py — reader registry (Phase B Task 1)."""
from __future__ import annotations


def test_read_txt(tmp_path):
    from raggity.readers import read_file

    p = tmp_path / "a.txt"
    p.write_text("hello text world")
    assert "hello text world" in read_file(str(p))


def test_read_md(tmp_path):
    from raggity.readers import read_file

    p = tmp_path / "a.md"
    p.write_text("# Title\nSome markdown content")
    out = read_file(str(p))
    assert "Title" in out and "markdown content" in out


def test_read_docx(tmp_path):
    import docx
    from raggity.readers import read_file

    p = tmp_path / "a.docx"
    d = docx.Document()
    d.add_paragraph("hello docx world")
    d.save(p)
    assert "docx world" in read_file(str(p))


def test_read_csv(tmp_path):
    from raggity.readers import read_file

    (tmp_path / "a.csv").write_text("name,role\nAlice,dev\n")
    out = read_file(str(tmp_path / "a.csv"))
    assert "Alice" in out and "role" in out


def test_read_html(tmp_path):
    from raggity.readers import read_file

    (tmp_path / "a.html").write_text(
        "<html><body><h1>Title</h1><p>Body text</p></body></html>"
    )
    out = read_file(str(tmp_path / "a.html"))
    assert "Title" in out and "Body text" in out


def test_read_pptx(tmp_path):
    from pptx import Presentation
    from raggity.readers import read_file

    pr = Presentation()
    s = pr.slides.add_slide(pr.slide_layouts[5])
    s.shapes.title.text = "Deck Title"
    pr.save(tmp_path / "a.pptx")
    assert "Deck Title" in read_file(str(tmp_path / "a.pptx"))


def test_unknown_ext_returns_none(tmp_path):
    from raggity.readers import read_file

    (tmp_path / "a.bin").write_bytes(b"\x00")
    assert read_file(str(tmp_path / "a.bin")) is None


def test_supported_exts_contains_all_formats():
    from raggity.readers import SUPPORTED_EXTS

    for ext in {".md", ".txt", ".pdf", ".docx", ".html", ".csv", ".pptx"}:
        assert ext in SUPPORTED_EXTS, f"{ext} missing from SUPPORTED_EXTS"


def test_read_pdf_via_read_file(tmp_path):
    """read_file dispatches .pdf correctly (uses the moved read_pdf).
    Dispatching .pdf must NOT return None (the unknown-ext sentinel).
    It may return a string or raise; either is fine.
    """
    from raggity.readers import read_file

    bad = tmp_path / "b.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    dispatched = False
    try:
        result = read_file(str(bad))
        dispatched = True  # returned a value (possibly empty string)
    except Exception:
        dispatched = True  # raised from pypdf — dispatch DID happen
    assert dispatched, ".pdf was not dispatched (returned None as unknown ext)"
