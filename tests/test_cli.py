import threading
import raggity.cli as cli_mod
import raggity.llm as llm_mod
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fix 1: serve --open opens browser via a background timer (not immediately)
# ---------------------------------------------------------------------------

def test_serve_open_uses_timer_not_direct_call(monkeypatch):
    """_open_browser_delayed schedules a Timer and does NOT call webbrowser.open inline."""
    timers = []
    opened = []

    class FakeTimer:
        def __init__(self, delay, fn, *a, args=(), kwargs=None, **kw):
            timers.append((delay, fn))
            self._fn = fn
        def start(self):
            pass  # don't actually fire

    monkeypatch.setattr("threading.Timer", FakeTimer)
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    # Import and call the helper introduced by the fix
    from raggity.cli import _open_browser_delayed
    _open_browser_delayed("http://127.0.0.1:8000", delay=1.0)

    assert len(timers) == 1, "exactly one Timer should be scheduled"
    assert timers[0][0] == 1.0, "delay should be 1.0s"
    assert len(opened) == 0, "webbrowser.open should NOT be called synchronously"


def test_serve_open_timer_is_daemon(monkeypatch):
    """Timer started by _open_browser_delayed must set daemon=True before start."""
    daemon_values = []

    class TrackingTimer:
        def __init__(self, delay, fn, *a, args=(), kwargs=None, **kw):
            self.daemon = False
        def start(self):
            daemon_values.append(self.daemon)

    monkeypatch.setattr("threading.Timer", TrackingTimer)
    monkeypatch.setattr("webbrowser.open", lambda url: None)

    from raggity.cli import _open_browser_delayed
    _open_browser_delayed("http://127.0.0.1:8000", delay=1.5)
    assert daemon_values == [True], "daemon must be True before start() is called"


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
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_query(prompt, options):
        # serve both expand variations and final answer via llm_mod.query
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
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    # All LLM calls (HyDE generation + final answer) go through llm_mod.query
    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--hyde"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_decompose_flag(tmp_path, monkeypatch):
    """CLI --decompose --plain should call ask_decompose and print the answer."""
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    # All LLM calls (decompose + answer) go through llm_mod.query
    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--decompose"])
    assert r.exit_code == 0 and "NAS" in r.stdout


def test_ask_decompose_overrides_other_transforms(tmp_path, monkeypatch):
    """--decompose combined with --expand prints override note."""
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])

    async def _fake_query(prompt, options):
        yield _AssistantMessage("Backups run nightly to the NAS [doc_1#00000000].")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    r = runner.invoke(cli_mod.app, ["ask", "how are backups done?", "--config", cfg,
                                    "--plain", "--decompose", "--expand"])
    assert r.exit_code == 0
    assert "NAS" in r.stdout
    # override note goes to stderr; typer.testing captures both in output
    assert "overrides" in r.output


def test_graph_build_exits_1_when_graph_off(tmp_path):
    """graph-build fails with exit code 1 when retrieval.graph=false (default)."""
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])
    r = runner.invoke(cli_mod.app, ["graph-build", "--config", cfg])
    assert r.exit_code == 1
    assert "graph" in r.output.lower()


def test_graph_build_exits_1_when_empty_index(tmp_path):
    """graph-build exits 1 with helpful message when index has no chunks."""
    notes = tmp_path / "notes"; notes.mkdir()
    cfg_path = tmp_path / "raggity.toml"
    cfg_path.write_text(
        f'[sources]\ninclude = ["{(notes / "*.md").as_posix()}"]\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
        f'[retrieval]\ngraph = true\n'
    )
    r = runner.invoke(cli_mod.app, ["graph-build", "--config", str(cfg_path)])
    assert r.exit_code == 1
    assert "empty" in r.output.lower() or "ingest" in r.output.lower()


def test_graph_build_succeeds(tmp_path, monkeypatch):
    """graph-build runs successfully with mocked LLM and prints 'Graph built.'"""
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbackups run nightly to the NAS")
    cfg_path = tmp_path / "raggity.toml"
    cfg_path.write_text(
        f'[sources]\ninclude = ["{(notes / "*.md").as_posix()}"]\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
        f'[retrieval]\ngraph = true\n'
    )
    # First ingest without graph (temporarily patch cfg to avoid LLM call during ingest)
    # We do: ingest → but graph=true means build_graph is called inside ingest.
    # So we monkeypatch LLM for both ingest AND graph-build.
    async def _fake_query(prompt, options):
        yield _AssistantMessage("E: NAS\nE: Backup System\nR: Backup System | writes to | NAS\n")
    monkeypatch.setattr(llm_mod, "query", _fake_query)
    monkeypatch.setattr(llm_mod, "AssistantMessage", _AssistantMessage)

    # Ingest first (graph=true so build_graph also runs inside ingest → OK with mock)
    runner.invoke(cli_mod.app, ["ingest", "--config", str(cfg_path)])

    r = runner.invoke(cli_mod.app, ["graph-build", "--config", str(cfg_path)])
    assert r.exit_code == 0
    assert "graph built" in r.output.lower()
    import os
    assert os.path.isfile(str(tmp_path / "idx" / "graph.json"))
