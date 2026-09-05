import tomllib

from typer.testing import CliRunner

from raggity.cli import app

runner = CliRunner()


def test_add_records_the_folder_and_ingests(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("hello", encoding="utf-8")
    cfg = tmp_path / "raggity.toml"
    called = []
    monkeypatch.setattr("raggity.cli._do_ingest", lambda c: called.append(c))

    res = runner.invoke(app, ["add", str(notes), "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert tomllib.loads(cfg.read_text(encoding="utf-8"))["sources"]["include"]
    assert called == [str(cfg)]


def test_no_ingest_records_without_indexing(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("hello", encoding="utf-8")
    cfg = tmp_path / "raggity.toml"
    monkeypatch.setattr(
        "raggity.cli._do_ingest",
        lambda c: (_ for _ in ()).throw(AssertionError("must not ingest")))

    res = runner.invoke(app, ["add", str(notes), "--config", str(cfg),
                              "--no-ingest"])
    assert res.exit_code == 0, res.output


def test_add_rejects_a_path_that_is_not_a_folder(tmp_path):
    res = runner.invoke(app, ["add", str(tmp_path / "nope")])
    assert res.exit_code != 0
    # Collapse whitespace before matching. rich wraps to the terminal width, and
    # CI's is narrower than a developer's: the phrase arrived as "is not a
    # \nfolder", so asserting on the raw output passed here and failed there.
    assert "not a folder" in " ".join(res.output.split())
