from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session", autouse=True)
def _make_sample_pdf():
    pdf = FIXTURES / "sample.pdf"
    if not pdf.exists():
        from pypdf import PdfWriter
        # Minimal 1-page PDF with extractable text is non-trivial to author by
        # hand; create a blank page and rely on text-PDF tests using .md/.txt.
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(pdf, "wb") as fh:
            writer.write(fh)
    yield
