import json

from typer.testing import CliRunner

from raggity.cli import app
from raggity.discover import Candidate

runner = CliRunner()


def test_json_output_is_machine_readable_and_never_prompts(tmp_path, monkeypatch):
    """rigma parses this. It must not prompt and must not print rich markup."""
    monkeypatch.setattr(
        "raggity.cli.cli_discover_source",
        lambda **kw: ([Candidate(tmp_path, "obsidian", 7, {".md": 7}, 100,
                                 "Obsidian vault - 7 notes")], True))
    res = runner.invoke(app, ["discover", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["complete"] is True
    assert data["candidates"][0]["kind"] == "obsidian"
    assert data["candidates"][0]["file_count"] == 7
    assert data["candidates"][0]["path"] == str(tmp_path)


def test_an_incomplete_scan_says_so_in_the_json(tmp_path, monkeypatch):
    """rigma needs to tell "nothing here" apart from "ran out of time"."""
    monkeypatch.setattr(
        "raggity.cli.cli_discover_source", lambda **kw: ([], False))
    res = runner.invoke(app, ["discover", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout) == {"complete": False, "candidates": []}
