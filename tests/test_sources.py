from pathlib import Path

import tomllib

from raggity.sources import add_sources, glob_for


def test_a_folder_becomes_a_recursive_glob():
    assert glob_for(Path("/home/u/notes")) == "/home/u/notes/**/*"


def test_creates_the_annotated_config_when_there_is_none(tmp_path):
    cfg = tmp_path / "raggity.toml"
    add_sources([tmp_path / "notes"], config=str(cfg))
    body = cfg.read_text(encoding="utf-8")
    assert "[sources]" in body
    assert "# raggity.toml" in body          # the template's comments survive
    assert tomllib.loads(body)["sources"]["include"] == [
        glob_for(tmp_path / "notes")]


def test_appending_keeps_existing_sources_and_comments(tmp_path):
    """The config is a file the user edits. Rewriting it from a model would
    silently drop every comment and every setting we do not know about."""
    cfg = tmp_path / "raggity.toml"
    cfg.write_text(
        '# my notes\n[sources]\ninclude = ["~/a/**/*"]\n\n'
        '[embedding]\nmodel = "custom"  # do not lose me\n',
        encoding="utf-8")
    add_sources([tmp_path / "b"], config=str(cfg))
    body = cfg.read_text(encoding="utf-8")
    assert "# my notes" in body and "# do not lose me" in body
    data = tomllib.loads(body)
    assert data["sources"]["include"] == ["~/a/**/*", glob_for(tmp_path / "b")]
    assert data["embedding"]["model"] == "custom"


def test_the_same_folder_twice_is_not_duplicated(tmp_path):
    cfg = tmp_path / "raggity.toml"
    add_sources([tmp_path / "notes"], config=str(cfg))
    add_sources([tmp_path / "notes"], config=str(cfg))
    assert len(tomllib.loads(cfg.read_text(encoding="utf-8"))
               ["sources"]["include"]) == 1
