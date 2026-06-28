"""Reader registry for raggity.

Dispatch table: Path.suffix.lower() -> reader function.
Optional-dep readers (docx, html, pptx) do a local import and raise a
friendly RuntimeError if the library is absent.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTS: set[str] = {".md", ".txt", ".pdf", ".docx", ".html", ".csv", ".pptx"}


# ---------------------------------------------------------------------------
# Per-type readers
# ---------------------------------------------------------------------------


def read_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_md(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_pdf(path: str) -> str:
    """Moved verbatim from loader.py."""
    from pypdf import PdfReader  # already a core dep

    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_docx(path: str) -> str:
    try:
        import docx  # python-docx  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("reading .docx needs: pip install raggity[docs]")
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs)


def read_html(path: str) -> str:
    try:
        from bs4 import BeautifulSoup  # beautifulsoup4  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("reading .html needs: pip install raggity[docs]")
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(" ", strip=True)


def read_csv(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(rows)


def read_pptx(path: str) -> str:
    try:
        from pptx import Presentation  # python-pptx  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("reading .pptx needs: pip install raggity[docs]")
    prs = Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, object] = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".html": read_html,
    ".csv": read_csv,
    ".pptx": read_pptx,
}


def read_file(path: str) -> str | None:
    """Read *path* and return its text content, or None for unknown extensions."""
    ext = Path(path).suffix.lower()
    fn = _DISPATCH.get(ext)
    if fn is None:
        return None
    return fn(path)  # type: ignore[operator]
