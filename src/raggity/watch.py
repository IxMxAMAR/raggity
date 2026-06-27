from __future__ import annotations
import os
import threading


class Debouncer:
    def __init__(self, interval: float, action) -> None:
        self._interval = interval
        self._action = action
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._interval, self._action)
            self._timer.daemon = True
            self._timer.start()


def _watch_dirs(globs: list[str]) -> list[str]:
    dirs = set()
    for g in globs:
        base = os.path.dirname(os.path.expanduser(g).split("*", 1)[0]) or "."
        dirs.add(base)
    return sorted(d for d in dirs if os.path.isdir(d))


def run_watch(rag, globs: list[str], debounce: float = 2.0):
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as exc:
        raise RuntimeError("watch needs extra deps: pip install raggity[watch]") from exc

    debouncer = Debouncer(debounce, lambda: rag.ingest())

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if not event.is_directory:
                debouncer.trigger()

    dirs = _watch_dirs(globs)
    observer = Observer()
    for d in dirs:
        observer.schedule(_Handler(), d, recursive=True)
    observer.start()
    return observer  # caller blocks/join; CLI joins until KeyboardInterrupt
