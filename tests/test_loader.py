from raggity.loader import load_documents, compute_file_hash


def test_loads_md_and_txt(fixtures_dir):
    docs = load_documents([str(fixtures_dir / "*.md"), str(fixtures_dir / "*.txt")])
    paths = {d.path.split("/")[-1].split("\\")[-1] for d in docs}
    assert "sample.md" in paths and "sample.txt" in paths


def test_title_from_h1(fixtures_dir):
    docs = load_documents([str(fixtures_dir / "sample.md")])
    assert docs[0].title == "Security Notes"


def test_title_from_filename_for_txt(fixtures_dir):
    docs = load_documents([str(fixtures_dir / "sample.txt")])
    assert docs[0].title == "sample"


def test_file_hash_stable(fixtures_dir):
    p = str(fixtures_dir / "sample.txt")
    assert compute_file_hash(p) == compute_file_hash(p)


def test_unsupported_extension_skipped(fixtures_dir, tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    docs = load_documents([str(tmp_path / "*.bin")])
    assert docs == []


def test_empty_file_skipped(tmp_path):
    from raggity.loader import load_documents
    (tmp_path / "empty.md").write_text("   \n")
    assert load_documents([str(tmp_path / "*.md")]) == []


def test_unreadable_pdf_skipped(tmp_path, caplog):
    from raggity.loader import load_documents
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    # must skip (not raise), returning no docs for the bad file
    docs = load_documents([str(tmp_path / "*.pdf")])
    assert all(not d.path.endswith("bad.pdf") for d in docs)


def test_document_path_is_posix(tmp_path):
    """Document.path must use forward slashes regardless of OS (POSIX form)."""
    (tmp_path / "notes.md").write_text("# My Notes\nContent here.")
    docs = load_documents([str(tmp_path / "*.md")])
    assert len(docs) == 1
    assert "\\" not in docs[0].path, (
        f"Document.path should be POSIX (no backslashes), got: {docs[0].path!r}"
    )
