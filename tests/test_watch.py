import time
from raggity.watch import Debouncer


def test_debouncer_coalesces_rapid_triggers():
    calls = {"n": 0}
    d = Debouncer(0.15, lambda: calls.__setitem__("n", calls["n"] + 1))
    for _ in range(5):
        d.trigger()
        time.sleep(0.02)
    time.sleep(0.3)
    assert calls["n"] == 1     # 5 rapid triggers → one action


def test_debouncer_runs_again_after_quiet():
    calls = {"n": 0}
    d = Debouncer(0.1, lambda: calls.__setitem__("n", calls["n"] + 1))
    d.trigger(); time.sleep(0.25)
    d.trigger(); time.sleep(0.25)
    assert calls["n"] == 2
