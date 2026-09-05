import tomllib

from raggity import setup_flow
from raggity.discover import Candidate


def test_never_prompts_without_a_tty(tmp_path, monkeypatch):
    """rigma runs these commands as a subprocess. A prompt here hangs a
    background ingest until its timeout, so this is load-bearing."""
    monkeypatch.setattr(setup_flow.sys.stdin, "isatty", lambda: False,
                        raising=False)
    monkeypatch.setattr(
        "raggity.setup_flow.discover",
        lambda **kw: (_ for _ in ()).throw(AssertionError("scanned anyway")))
    assert setup_flow.offer(str(tmp_path / "raggity.toml")) is False


def test_env_var_forces_non_interactive(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_flow.sys.stdin, "isatty", lambda: True,
                        raising=False)
    monkeypatch.setenv("RAGGITY_NONINTERACTIVE", "1")
    monkeypatch.setattr(
        "raggity.setup_flow.discover",
        lambda **kw: (_ for _ in ()).throw(AssertionError("scanned anyway")))
    assert setup_flow.offer(str(tmp_path / "raggity.toml")) is False


def _interactive(monkeypatch):
    monkeypatch.setattr(setup_flow.sys.stdin, "isatty", lambda: True,
                        raising=False)
    monkeypatch.delenv("RAGGITY_NONINTERACTIVE", raising=False)


def test_choosing_a_candidate_writes_it_and_reports_success(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    cfg = tmp_path / "raggity.toml"
    _interactive(monkeypatch)
    monkeypatch.setattr(
        "raggity.setup_flow.discover",
        lambda **kw: ([Candidate(notes, "known", 3, {".md": 3}, 50, "3 notes")],
                      True))
    monkeypatch.setattr("raggity.setup_flow._read_choice", lambda *a, **k: "1")
    monkeypatch.setattr("raggity.setup_flow._confirm", lambda *a, **k: False)

    assert setup_flow.offer(str(cfg)) is True
    assert tomllib.loads(cfg.read_text(encoding="utf-8"))["sources"]["include"]


def test_quitting_writes_nothing(tmp_path, monkeypatch):
    cfg = tmp_path / "raggity.toml"
    _interactive(monkeypatch)
    monkeypatch.setattr(
        "raggity.setup_flow.discover",
        lambda **kw: ([Candidate(tmp_path, "known", 3, {".md": 3}, 50, "3")], True))
    monkeypatch.setattr("raggity.setup_flow._read_choice", lambda *a, **k: "q")

    assert setup_flow.offer(str(cfg)) is False
    assert not cfg.exists()


def test_a_large_folder_is_confirmed_separately(tmp_path, monkeypatch):
    """Above the threshold the first ingest is minutes, so it says the number
    and defaults to no."""
    big = tmp_path / "big"
    big.mkdir()
    cfg = tmp_path / "raggity.toml"
    asked = []
    _interactive(monkeypatch)
    monkeypatch.setattr(
        "raggity.setup_flow.discover",
        lambda **kw: ([Candidate(big, "known", 40000, {".md": 40000}, 50, "many")],
                      True))
    answers = iter(["1", "q"])
    monkeypatch.setattr("raggity.setup_flow._read_choice",
                        lambda *a, **k: next(answers))

    def _confirm(msg, default=True):
        asked.append((msg, default))
        return False

    monkeypatch.setattr("raggity.setup_flow._confirm", _confirm)
    setup_flow.offer(str(cfg))
    assert any("40,000" in m and d is False for m, d in asked)
