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
