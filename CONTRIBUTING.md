# Contributing to raggity

Thank you for your interest in contributing!

## Dev setup

```bash
git clone https://github.com/IxMxAMAR/raggity.git
cd raggity

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -e ".[server,qdrant,openai,web,docs,ocr,graph,otel,dev]"
```

## Running tests

```bash
python -m pytest -q
```

All LLM calls in the test suite are **mocked** — tests run fully offline with no API key required. No `ANTHROPIC_API_KEY`, no network, no model downloads needed to run the suite (fastembed model weights are the only download, cached after the first run).

To run a specific file or test:

```bash
python -m pytest tests/test_retrieval.py -q
python -m pytest -k "test_ask" -q
```

## Code style

- Formatter: `black` (via `pre-commit`). Run `pre-commit install` after cloning to enable automatic formatting on commit.
- Linter: `ruff`.
- Type annotations encouraged on public APIs.
- Keep new tests alongside new features — PRs without tests for new behaviour will be asked to add coverage.

## PR process

1. Fork the repo and create a branch from `master`.
2. Make your changes; add or update tests as needed.
3. Ensure `python -m pytest -q` passes with 0 failures and 0 warnings.
4. Open a pull request against `master`. Fill in the PR template.
5. A maintainer will review and merge.

For large changes, open an issue first to discuss the design.

## Commit messages

Use conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `perf:` prefixes.

## Reporting issues

Use the GitHub issue templates — bug report or feature request. Include raggity version (`rag --version`), Python version, and OS.
