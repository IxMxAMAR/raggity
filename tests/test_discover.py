from pathlib import Path

from raggity.discover import DISCOVER_EXTS, Candidate, count_indexable, scan_named


def _tree(root: Path, names: list[str]) -> Path:
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return root


def test_images_are_not_counted_as_indexable():
    """Image formats need the OCR extra, so offering a photo folder as
    documents would be a lie about what ingest would take."""
    assert ".md" in DISCOVER_EXTS and ".pdf" in DISCOVER_EXTS
    assert ".png" not in DISCOVER_EXTS and ".jpg" not in DISCOVER_EXTS


def test_count_indexable_reports_totals_and_a_breakdown(tmp_path):
    _tree(tmp_path, ["a.md", "b.md", "c.pdf", "ignore.exe", "sub/d.md"])
    total, exts = count_indexable(tmp_path)
    assert total == 4
    assert exts == {".md": 3, ".pdf": 1}


def test_scan_named_skips_roots_with_nothing_to_index(tmp_path):
    empty = tmp_path / "Empty"
    empty.mkdir()
    notes = _tree(tmp_path / "Notes", ["a.md", "b.md"])
    got = scan_named([empty, notes, tmp_path / "DoesNotExist"])
    assert [c.path for c in got] == [notes]
    assert isinstance(got[0], Candidate)
    assert got[0].file_count == 2 and got[0].kind == "known"


def test_scan_named_never_reads_file_contents(tmp_path, monkeypatch):
    """Discovery is metadata only. A user's documents are not opened to
    decide whether to offer the folder they live in."""
    _tree(tmp_path, ["secret.md"])
    real_open = open

    def _boom(path, *a, **kw):
        if str(path).endswith(".md"):
            raise AssertionError(f"discovery opened {path}")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", _boom)
    assert scan_named([tmp_path])[0].file_count == 1
