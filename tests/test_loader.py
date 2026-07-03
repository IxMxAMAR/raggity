import logging

from raggity.loader import load_documents, compute_file_hash, scan_sources


def test_loads_md_and_txt(fixtures_dir):
    docs, _, _ = load_documents([str(fixtures_dir / "*.md"), str(fixtures_dir / "*.txt")])
    paths = {d.path.split("/")[-1].split("\\")[-1] for d in docs}
    assert "sample.md" in paths and "sample.txt" in paths


def test_title_from_h1(fixtures_dir):
    docs, _, _ = load_documents([str(fixtures_dir / "sample.md")])
    assert docs[0].title == "Security Notes"


def test_title_from_filename_for_txt(fixtures_dir):
    docs, _, _ = load_documents([str(fixtures_dir / "sample.txt")])
    assert docs[0].title == "sample"


def test_file_hash_stable(fixtures_dir):
    p = str(fixtures_dir / "sample.txt")
    assert compute_file_hash(p) == compute_file_hash(p)


def test_unsupported_extension_skipped(fixtures_dir, tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    docs, _, _ = load_documents([str(tmp_path / "*.bin")])
    assert docs == []


def test_empty_file_skipped(tmp_path):
    from raggity.loader import load_documents
    (tmp_path / "empty.md").write_text("   \n")
    docs, _, _ = load_documents([str(tmp_path / "*.md")])
    assert docs == []


def test_unreadable_pdf_skipped(tmp_path, caplog):
    from raggity.loader import load_documents
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    # must skip (not raise), returning no docs for the bad file
    docs, _, _ = load_documents([str(tmp_path / "*.pdf")])
    assert all(not d.path.endswith("bad.pdf") for d in docs)


def test_document_path_is_posix(tmp_path):
    """Document.path must use forward slashes regardless of OS (POSIX form)."""
    (tmp_path / "notes.md").write_text("# My Notes\nContent here.")
    docs, _, _ = load_documents([str(tmp_path / "*.md")])
    assert len(docs) == 1
    assert "\\" not in docs[0].path, (
        f"Document.path should be POSIX (no backslashes), got: {docs[0].path!r}"
    )


def test_mixed_md_and_docx_folder(tmp_path):
    """A folder with .md and .docx files loads both document types."""
    import docx as _docx

    # Write a markdown file
    (tmp_path / "notes.md").write_text("# Meeting Notes\nDecision: ship it.")

    # Write a docx file
    d = _docx.Document()
    d.add_paragraph("Contract text here")
    d.save(tmp_path / "contract.docx")

    docs, _, _ = load_documents([str(tmp_path / "*.md"), str(tmp_path / "*.docx")])
    filenames = {d.path.rsplit("/", 1)[-1] for d in docs}

    assert "notes.md" in filenames, "markdown file not loaded"
    assert "contract.docx" in filenames, "docx file not loaded"

    texts = {d.path.rsplit("/", 1)[-1]: d.text for d in docs}
    assert "Decision: ship it." in texts["notes.md"]
    assert "Contract text" in texts["contract.docx"]


def test_missing_dep_file_lands_in_skipped_needs_extra(tmp_path, monkeypatch):
    """A file whose reader raises MissingDependencyError is counted in skipped_needs_extra."""
    import raggity.readers as r
    from raggity.readers import MissingDependencyError

    # Create a .png file — image reader will call _run_ocr
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Monkeypatch _run_ocr to simulate missing ocr extra
    monkeypatch.setattr(r, "_run_ocr", lambda src: (_ for _ in ()).throw(
        MissingDependencyError(extra="ocr", message="needs ocr")
    ))

    docs, skipped_needs_extra, skipped_generic = load_documents([str(tmp_path / "*.png")])
    assert docs == []
    assert skipped_needs_extra.get("ocr", 0) == 1
    assert skipped_generic == 0


def test_junk_dir_below_prefix_is_pruned(tmp_path):
    """A junk dir (AppData) appearing BELOW the include pattern's static prefix
    is pruned; a real notes dir at the same level is kept."""
    (tmp_path / "AppData").mkdir()
    (tmp_path / "AppData" / "junk.txt").write_text("junk from appdata")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "real.txt").write_text("real note")
    # static prefix is tmp_path; AppData/notes are segments below it
    docs, _, _ = load_documents([str(tmp_path / "**" / "*.txt")])
    names = {d.path.rsplit("/", 1)[-1] for d in docs}
    assert names == {"real.txt"}


def test_junk_name_at_or_above_prefix_is_kept(tmp_path):
    """A pattern pointing INSIDE a junk-named dir (junk name is at/above the
    static prefix) still ingests — pruning only looks BELOW the prefix."""
    appdata = tmp_path / "AppData" / "notes"
    appdata.mkdir(parents=True)
    (appdata / "real.md").write_text("# Real\n\nkept because prefix reaches here")
    docs, _, _ = load_documents([str(appdata / "*.md")])
    names = {d.path.rsplit("/", 1)[-1] for d in docs}
    assert names == {"real.md"}


def test_raggity_index_dir_never_swept(tmp_path):
    """The .raggity index dir is a built-in junk exclusion below the prefix."""
    (tmp_path / ".raggity").mkdir()
    (tmp_path / ".raggity" / "cached.md").write_text("# X\n\nindex internals")
    (tmp_path / "keep.md").write_text("# Keep\n\nreal content")
    docs, _, _ = load_documents([str(tmp_path / "**" / "*.md")])
    names = {d.path.rsplit("/", 1)[-1] for d in docs}
    assert names == {"keep.md"}


def test_user_exclude_glob_honored(tmp_path):
    """A user exclude glob (fnmatch on the posix path) skips matching files."""
    (tmp_path / "keep.md").write_text("# Keep\n\nreal")
    (tmp_path / "draft.md").write_text("# Draft\n\nskip me")
    docs, _, _ = load_documents(
        [str(tmp_path / "*.md")], exclude=["**/draft.md"]
    )
    names = {d.path.rsplit("/", 1)[-1] for d in docs}
    assert names == {"keep.md"}


def test_scanned_count_populated(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nx")
    (tmp_path / "b.txt").write_text("y")
    scan = scan_sources([str(tmp_path / "*.md"), str(tmp_path / "*.txt")], {})
    assert scan.scanned == 2


def test_skip_paths_emit_at_info_not_warning(tmp_path, caplog):
    """Per-file empty/unsupported skips log at INFO, not WARNING (quiet console)."""
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    with caplog.at_level(logging.INFO, logger="raggity.loader"):
        load_documents([str(tmp_path / "*.md"), str(tmp_path / "*.bin")])
    records = [r for r in caplog.records if "skipping" in r.message]
    assert records, "expected at least one skip log"
    assert all(r.levelno == logging.INFO for r in records), (
        f"skip logs must be INFO, got {[r.levelname for r in records]}"
    )


def test_generic_error_lands_in_skipped_generic(tmp_path, monkeypatch):
    """A file that raises a plain exception is counted in skipped_generic."""
    import raggity.readers as r

    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 corrupt")
    # Force read_pdf to raise a generic error
    monkeypatch.setattr(r, "_pdf_text", lambda p: (_ for _ in ()).throw(ValueError("corrupt")))
    monkeypatch.setattr(r, "_ocr_pdf", lambda p: (_ for _ in ()).throw(ValueError("corrupt")))

    docs, skipped_needs_extra, skipped_generic = load_documents([str(tmp_path / "*.pdf")])
    assert docs == []
    assert skipped_generic >= 1
    assert not skipped_needs_extra
