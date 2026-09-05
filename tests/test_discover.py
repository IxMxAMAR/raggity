import json
from pathlib import Path

from raggity.discover import (
    DISCOVER_EXTS,
    Candidate,
    count_indexable,
    discover,
    scan_named,
    scan_obsidian,
)


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


def test_obsidian_vaults_are_read_from_obsidian_s_own_config(tmp_path, monkeypatch):
    """The highest-confidence signal on the machine, and it costs one file
    read rather than a search."""
    vault = _tree(tmp_path / "Writing", ["ch1.md", "ch2.md"])
    (vault / ".obsidian").mkdir()
    cfg = tmp_path / "obsidian.json"
    cfg.write_text(json.dumps(
        {"vaults": {"abc": {"path": str(vault)}}}), encoding="utf-8")
    monkeypatch.setattr("raggity.discover.obsidian_config_path", lambda: cfg)

    got = scan_obsidian()
    assert [c.path for c in got] == [vault]
    assert got[0].kind == "obsidian"
    assert got[0].confidence > 50           # outranks a plain named folder
    assert "Obsidian vault" in got[0].why


def test_a_vault_that_no_longer_exists_is_skipped(tmp_path, monkeypatch):
    """Obsidian keeps deleted vaults in its list; offering one is a dead end."""
    cfg = tmp_path / "obsidian.json"
    cfg.write_text(json.dumps(
        {"vaults": {"gone": {"path": str(tmp_path / "Deleted")}}}),
        encoding="utf-8")
    monkeypatch.setattr("raggity.discover.obsidian_config_path", lambda: cfg)
    assert scan_obsidian() == []


def test_a_corrupt_obsidian_config_is_not_fatal(tmp_path, monkeypatch):
    cfg = tmp_path / "obsidian.json"
    cfg.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("raggity.discover.obsidian_config_path", lambda: cfg)
    assert scan_obsidian() == []


def test_a_dense_folder_deep_in_home_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr("raggity.discover.scan_obsidian", lambda *a: [])
    _tree(tmp_path, [f"projects/novel/notes/ch{i}.md" for i in range(6)])
    cands, complete = discover(home=tmp_path, cwd=tmp_path / "elsewhere")
    assert complete
    assert tmp_path / "projects" / "novel" / "notes" in [c.path for c in cands]


def test_only_the_shallowest_folder_in_a_chain_is_offered(tmp_path, monkeypatch):
    """A vault should appear once, not once per subfolder."""
    monkeypatch.setattr("raggity.discover.scan_obsidian", lambda *a: [])
    _tree(tmp_path, [f"vault/a{i}.md" for i in range(6)]
                    + [f"vault/sub/b{i}.md" for i in range(6)])
    cands, _ = discover(home=tmp_path, cwd=tmp_path / "elsewhere")
    found = [c.path for c in cands if c.kind == "found"]
    assert tmp_path / "vault" in found
    assert tmp_path / "vault" / "sub" not in found


def test_a_sparse_folder_is_not_offered(tmp_path, monkeypatch):
    """Under five files is noise, not a document folder."""
    monkeypatch.setattr("raggity.discover.scan_obsidian", lambda *a: [])
    _tree(tmp_path, ["stray/one.md", "stray/two.md"])
    cands, _ = discover(home=tmp_path, cwd=tmp_path / "elsewhere")
    assert tmp_path / "stray" not in [c.path for c in cands]


def test_running_out_of_budget_is_reported_not_hidden(tmp_path, monkeypatch):
    """`complete=False` is what stops "found nothing" being confused with
    "ran out of time" — the user is offered a deeper scan instead."""
    monkeypatch.setattr("raggity.discover.scan_obsidian", lambda *a: [])
    _tree(tmp_path, [f"d{i}/f{j}.md" for i in range(30) for j in range(6)])
    # A negative budget, not 0.0: `monotonic() + 0.0` is not reliably in the
    # past on Windows' ~15ms clock, so the zero case passed or failed by luck.
    cands, complete = discover(home=tmp_path, cwd=tmp_path / "elsewhere",
                               budget_s=-1.0)
    assert complete is False


def test_candidates_come_back_best_first(tmp_path, monkeypatch):
    vault = _tree(tmp_path / "V", ["a.md"])
    monkeypatch.setattr(
        "raggity.discover.scan_obsidian",
        lambda *a: [Candidate(vault, "obsidian", 1, {".md": 1}, 100,
                              "Obsidian vault")])
    _tree(tmp_path / "Documents", [f"d{i}.md" for i in range(6)])
    cands, _ = discover(home=tmp_path, cwd=tmp_path / "elsewhere")
    assert cands[0].kind == "obsidian"
