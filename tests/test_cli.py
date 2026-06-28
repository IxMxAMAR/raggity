import raggity.cli as cli_mod
import raggity.llm as llm_mod
from typer.testing import CliRunner

runner = CliRunner()


class _Block:
    def __init__(self, text): self.text = text


class _AssistantMessage:
    def __init__(self, text): self.content = [_Block(text)]


def _make_config(tmp_path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg = tmp_path / "raggity.toml"
    cfg.write_text(
        f'[sources]\ninclude = ["{(notes / "*.md").as_posix()}"]\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )
    return str(cfg)


def test_ingest_then_status(tmp_path):
    cfg = _make_config(tmp_path)
    r1 = runner.invoke(cli_mod.app, ["ingest", "--config", cfg])
    assert r1.exit_code == 0
    r2 = runner.invoke(cli_mod.app, ["status", "--config", cfg])
    assert r2.exit_code == 0 and "chunks" in r2.stdout.lower()


def test_ask_expand_flag(tmp_path, monkeypatch):
    import raggity.query_transform as qt
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_expand(prompt, options):
        yield _AssistantMessage("backups overview\nNAS schedule")
    monkeypatch.setattr(qt, "query", _fake_expand)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg, "--plain", "--expand"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_streams_by_default(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)
    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_plain(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")

    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg, "--plain"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_hyde_flag(tmp_path, monkeypatch):
    """CLI --hyde --plain should work and print the answer."""
    import raggity.query_transform as qt
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_qt(prompt, options):
        yield _AssistantMessage("Backups are stored on a NAS device nightly.")
    monkeypatch.setattr(qt, "query", _fake_qt)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)

    async def _fake_answer(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_answer)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--hyde"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_decompose_flag(tmp_path, monkeypatch):
    """CLI --decompose --plain should call ask_decompose and print the answer."""
    import raggity.query_transform as qt
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_decomp(prompt, options):
        yield _AssistantMessage("how often?\nwhere stored?")
    monkeypatch.setattr(qt, "query", _fake_decomp)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)

    async def _fake_answer(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_answer)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--decompose"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_decompose_overrides_other_transforms(tmp_path, monkeypatch):
    """--decompose combined with --expand prints override note."""
    import raggity.query_transform as qt
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_decomp(prompt, options):
        yield _AssistantMessage("how often?\nwhere stored?")
    monkeypatch.setattr(qt, "query", _fake_decomp)
    monkeypatch.setattr(qt, "AssistantMessage", _AssistantMessage)

    async def _fake_answer(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_answer)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--decompose", "--expand"])
    assert r.exit_code == 0
    assert "NAS" in r.stdout
    # override note goes to stderr; typer.testing captures both in output
    assert "overrides" in r.output
