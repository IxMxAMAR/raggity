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


def test_ingest_summary_shows_skipped_count(tmp_path):
    """The 'Indexed.' line reports the skipped_generic count."""
    cfg = _make_config(tmp_path)
    r = runner.invoke(cli_mod.app, ["ingest", "--config", cfg])
    assert r.exit_code == 0
    assert "skipped=" in r.output


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


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import raggity
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        expected = tomllib.load(fh)["project"]["version"]
    assert raggity.__version__ == expected


def test_status_empty_kb_shows_hint(tmp_path):
    """status with 0 chunks appends the init hint."""
    notes = tmp_path / "notes"; notes.mkdir()
    cfg_path = tmp_path / "raggity.toml"
    cfg_path.write_text(
        f'[sources]\ninclude = ["{(notes / "*.md").as_posix()}"]\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )
    # Ingest nothing → 0 chunks
    runner.invoke(cli_mod.app, ["ingest", "--config", str(cfg_path)])
    r = runner.invoke(cli_mod.app, ["status", "--config", str(cfg_path)])
    assert r.exit_code == 0
    assert "rag init" in r.output or "rag ingest" in r.output


def test_status_populated_kb_no_hint(tmp_path):
    """status with chunks present must NOT show the empty-KB hint."""
    cfg = _make_config(tmp_path)
    runner.invoke(cli_mod.app, ["ingest", "--config", cfg])
    r = runner.invoke(cli_mod.app, ["status", "--config", cfg])
    assert r.exit_code == 0
    assert "rag init" not in r.output


def test_ask_empty_kb_shows_hint_and_exits_0(tmp_path):
    """ask with 0 chunks prints the hint and exits cleanly."""
    notes = tmp_path / "notes"; notes.mkdir()
    cfg_path = tmp_path / "raggity.toml"
    cfg_path.write_text(
        f'[sources]\ninclude = ["{(notes / "*.md").as_posix()}"]\n'
        f'[index]\npath = "{(tmp_path / "idx").as_posix()}"\n'
    )
    runner.invoke(cli_mod.app, ["ingest", "--config", str(cfg_path)])
    r = runner.invoke(cli_mod.app, ["ask", "anything", "--config", str(cfg_path), "--plain"])
    assert r.exit_code == 0
    assert "rag init" in r.output or "rag ingest" in r.output


def test_ask_no_config_file_shows_hint(tmp_path, monkeypatch):
    """ask with no raggity.toml anywhere prints no-config hint."""
    monkeypatch.chdir(tmp_path)  # no raggity.toml here
    # Patch platformdirs so user config dir is also tmp_path (no toml there either)
    import platformdirs
    monkeypatch.setattr(platformdirs, "user_config_dir", lambda *a, **kw: str(tmp_path))
    r = runner.invoke(cli_mod.app, ["ask", "anything", "--plain"])
    # Should exit 0 and show a no-config hint (not crash)
    assert r.exit_code == 0
    assert "rag init" in r.output


def test_ingest_no_config_file_shows_hint(tmp_path, monkeypatch):
    """ingest with no raggity.toml prints the init hint."""
    monkeypatch.chdir(tmp_path)
    import platformdirs
    monkeypatch.setattr(platformdirs, "user_config_dir", lambda *a, **kw: str(tmp_path))
    r = runner.invoke(cli_mod.app, ["ingest"])
    # exit code doesn't matter (may still run with defaults); the hint must appear
    assert "rag init" in r.output


def test_init_creates_toml(tmp_path, monkeypatch):
    """rag init writes a raggity.toml template when none exists."""
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(cli_mod.app, ["init"])
    assert r.exit_code == 0
    cfg = tmp_path / "raggity.toml"
    assert cfg.exists(), "raggity.toml was not created"
    content = cfg.read_text()
    assert "[sources]" in content
    assert "include" in content
    assert "[generation]" in content
    # Next-steps guidance printed
    assert "rag ingest" in r.output
    # profile preset documented as a commented-out line (parses fine as-is)
    assert '# profile = "low-ram"' in content
    import tomllib
    tomllib.loads(content)


def test_init_refuses_to_overwrite(tmp_path, monkeypatch):
    """rag init does not clobber an existing raggity.toml."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "raggity.toml"
    cfg.write_text("# existing\n")
    r = runner.invoke(cli_mod.app, ["init"])
    assert r.exit_code == 0
    assert cfg.read_text() == "# existing\n", "existing file was overwritten"
    assert "already exists" in r.output.lower()


def test_init_custom_path(tmp_path):
    """rag init --config <path> writes to that path."""
    dest = tmp_path / "custom.toml"
    r = runner.invoke(cli_mod.app, ["init", "--config", str(dest)])
    assert r.exit_code == 0
    assert dest.exists()
    assert "[sources]" in dest.read_text()


# ---------------------------------------------------------------------------
# rag model - switch generation backend/model (v0.10.0)
# ---------------------------------------------------------------------------

def test_model_no_arg_shows_current(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nbackend = "ollama"\nmodel = "gemma3"\n'
                    'base_url = "http://localhost:11434/v1"\n')
    r = runner.invoke(cli_mod.app, ["model", "--config", str(dest)])
    assert r.exit_code == 0
    assert "backend=ollama" in r.output and "model=gemma3" in r.output
    assert "base_url=" in r.output


def test_model_switch_edits_model_and_backend(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "claude-opus-4-8"\n')
    r = runner.invoke(cli_mod.app, ["model", "gemma3", "-p", "ollama", "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["model"] == "gemma3"
    assert data["generation"]["backend"] == "ollama"
    assert "ollama pull gemma3" in r.output  # ollama hint


def test_model_anthropic_alias_maps_to_claude(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "claude-opus-4-8", "-p", "anthropic",
                                    "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    assert tomllib.loads(dest.read_text())["generation"]["backend"] == "claude"


def test_model_model_only_leaves_backend(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nbackend = "ollama"\nmodel = "old"\n')
    r = runner.invoke(cli_mod.app, ["model", "newmodel", "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["model"] == "newmodel"
    assert data["generation"]["backend"] == "ollama"  # unchanged


def test_model_preserves_template_comments(tmp_path, monkeypatch):
    """Editing a comment-rich init template must keep its comments (tomlkit)."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli_mod.app, ["init"])
    dest = tmp_path / "raggity.toml"
    assert "# Edit [sources] then run" in dest.read_text()  # sanity: comment present
    r = runner.invoke(cli_mod.app, ["model", "gemma3", "-p", "ollama"])
    assert r.exit_code == 0
    after = dest.read_text()
    assert "# Edit [sources] then run" in after  # comment survived the edit
    import tomllib
    assert tomllib.loads(after)["generation"]["model"] == "gemma3"


def test_model_external_keeps_existing_base_url(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "old"\nbase_url = "http://127.0.0.1:9999/v1"\n')
    r = runner.invoke(cli_mod.app, ["model", "rigma-model", "-p", "external",
                                    "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["backend"] == "external"
    assert data["generation"]["model"] == "rigma-model"
    assert data["generation"]["base_url"] == "http://127.0.0.1:9999/v1"


def test_model_external_accepts_base_url_option(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "rigma-model", "-p", "external",
                                    "--base-url", "http://127.0.0.1:9999",
                                    "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["backend"] == "external"
    assert data["generation"]["base_url"] == "http://127.0.0.1:9999"


def test_model_external_missing_base_url_errors(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "rigma-model", "-p", "external",
                                    "--config", str(dest)])
    assert r.exit_code != 0
    assert "base_url" in r.output or "base-url" in r.output


def test_model_base_url_option_overrides_for_any_provider(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "gemma3", "-p", "ollama",
                                    "--base-url", "http://box:11434/v1",
                                    "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    assert tomllib.loads(dest.read_text())["generation"]["base_url"] == "http://box:11434/v1"


def test_model_missing_file_creates_then_applies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import platformdirs
    monkeypatch.setattr(platformdirs, "user_config_dir", lambda *a, **kw: str(tmp_path / "nope"))
    r = runner.invoke(cli_mod.app, ["model", "gemma3", "-p", "ollama"])
    assert r.exit_code == 0
    assert "Created" in r.output
    dest = tmp_path / "raggity.toml"
    assert dest.exists()
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["model"] == "gemma3"
    assert data["generation"]["backend"] == "ollama"


def test_model_invalid_provider_exits_1(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "m", "-p", "bogus", "--config", str(dest)])
    assert r.exit_code == 1
    assert "Invalid provider" in r.output


def test_model_local_alias_writes_openai_backend_and_base_url(tmp_path):
    dest = tmp_path / "raggity.toml"
    dest.write_text('[generation]\nmodel = "x"\n')
    r = runner.invoke(cli_mod.app, ["model", "qwen2.5", "-p", "lmstudio", "--config", str(dest)])
    assert r.exit_code == 0
    import tomllib
    data = tomllib.loads(dest.read_text())
    assert data["generation"]["backend"] == "openai"
    assert data["generation"]["base_url"] == "http://localhost:1234/v1"


def test_model_list_renders_table(monkeypatch):
    import raggity.providers as prov
    fake = [
        prov.ProviderStatus(name="ollama", base_url="http://localhost:11434/v1",
                            running=True, installed=True, models=["gemma3"], auto_startable=True),
        prov.ProviderStatus(name="lmstudio", base_url="http://localhost:1234/v1",
                            running=False, installed=False, models=[], auto_startable=False),
    ]
    monkeypatch.setattr(prov, "discover", lambda: fake)
    r = runner.invoke(cli_mod.app, ["model", "--list"])
    assert r.exit_code == 0
    assert "ollama" in r.output and "lmstudio" in r.output
    assert "gemma3" in r.output
