"""PyInstaller entry point for the raggity `rag` CLI.

The console script `rag = raggity.cli:app` points at a Typer *object*, not a
callable. PyInstaller needs a module with a real `__main__` block, so this thin
shim invokes the app. Keep it dependency-free beyond raggity itself.
"""
from raggity.cli import app

if __name__ == "__main__":
    app()
