from __future__ import annotations

import shutil
import threading
import typer
from rich.console import Console

from .config import load_config, _find_config_path
from .core import Raggity, _run_async
from .evaluate import evaluate, load_golden


def _open_browser_delayed(url: str, delay: float = 1.2) -> None:
    """Schedule a background timer that opens *url* after *delay* seconds.

    Opening from a timer (not inline) lets uvicorn finish binding before the
    browser sends its first request, avoiding a connection-refused tab.
    """
    import webbrowser

    t = threading.Timer(delay, webbrowser.open, args=(url,))
    t.daemon = True
    t.start()

app = typer.Typer(help="raggity - local-first RAG over your notes, answered by Claude.")
console = Console()

_EMPTY_KB_HINT = (
    "[yellow]Knowledge base is empty.[/yellow] "
    r"Run [cyan]rag init[/cyan], set \[sources] include patterns, "
    "then [cyan]rag ingest[/cyan]."
)
_NO_CONFIG_HINT = (
    "[yellow]No raggity.toml found.[/yellow] "
    "Run [cyan]rag init[/cyan] to create one."
)


def _check_no_config(config: str | None) -> bool:
    """Return True (and print hint) when no config file is found."""
    if _find_config_path(config) is None:
        console.print(_NO_CONFIG_HINT)
        return True
    return False


def _rag(config: str | None) -> Raggity:
    return Raggity(load_config(config))


_INIT_TEMPLATE = """\
# raggity.toml - configuration for raggity
# Edit [sources] then run: rag ingest

[sources]
# Glob patterns for files to index. Supports **, *, and ~ expansion.
include = [
  "**/*.md",
  "**/*.txt",
  "**/*.pdf",
  "**/*.docx",
  # "**/*.html",     # included in base install
  # "**/*.pptx",     # included in base install
  # "**/*.png",      # requires: pip install raggity[ocr]
  # "**/*.jpg",      # requires: pip install raggity[ocr]
]
# Extra glob patterns to skip (matched against each file's posix path), e.g.
#   exclude = ["**/drafts/**", "**/*.tmp.md"]
# Built-in junk dirs are ALWAYS pruned below your include prefixes:
#   AppData, node_modules, .git, __pycache__, site-packages, .venv, venv,
#   dist-packages, .raggity, .npm, .nuget, .gradle, .cargo, .conda
exclude = []

[embedding]
# Model used to embed your documents. CPU-friendly default.
model = "BAAI/bge-small-en-v1.5"
provider = "cpu"

[retrieval]
hybrid = true    # combine dense + sparse (BM25) retrieval
rerank = true    # cross-encoder reranking for precision
top_k = 5        # chunks passed to the LLM

[generation]
# auth = "auto"   -- uses ANTHROPIC_API_KEY if set, otherwise claude login session
# Run `claude login` once if you have a Claude subscription.
# Or: export ANTHROPIC_API_KEY=sk-ant-...
auth = "auto"
model = "claude-opus-4-8"
# Opt-in personalization (off by default). persona: free-form text appended to
# the system prompt as user context; personal_kb: treat the KB as the current
# user's own (first-person docs/questions refer to them). Grounding rules still apply.
# persona = ""
# personal_kb = false

[index]
path = ".raggity/index"
"""


@app.command()
def init(config: str = typer.Option(None, "--config")):
    """Write an annotated raggity.toml template (does not overwrite)."""
    from pathlib import Path as _Path  # noqa: PLC0415
    dest = _Path(config) if config else _Path.cwd() / "raggity.toml"
    if dest.exists():
        console.print(f"[yellow]{dest.name} already exists - not overwriting.[/yellow]")
        console.print(f"  Edit {dest} directly, then run [cyan]rag ingest[/cyan].")
        return
    dest.write_text(_INIT_TEMPLATE, encoding="utf-8")
    console.print(f"[green]Created[/green] {dest}")
    console.print("\nNext steps:")
    console.print(r"  1. Edit [cyan]raggity.toml[/cyan] - set \[sources] include patterns")
    console.print("  2. Run [cyan]rag ingest[/cyan]  - index your files")
    console.print('  3. Run [cyan]rag ask "your question here"[/cyan]')


def _print_provider_table(statuses) -> None:
    """Render discovered local providers as a Rich table."""
    from rich.table import Table  # noqa: PLC0415
    table = Table(title="Local LLM providers")
    table.add_column("provider")
    table.add_column("status")
    table.add_column("models")
    for st in statuses:
        if st.running:
            status = "[green]running[/green]"
        elif st.installed:
            status = "[yellow]installed (not running)[/yellow]"
        else:
            status = "[dim]not found[/dim]"
        if st.auto_startable and not st.running:
            status += " [dim](auto-startable)[/dim]"
        models = ", ".join(st.models[:6]) if st.models else "-"
        if st.models and len(st.models) > 6:
            models += ", ..."
        table.add_row(st.name, status, models)
    console.print(table)


@app.command()
def model(
    model_name: str = typer.Argument(None, help="Model name to switch to (omit to show current)."),
    provider: str = typer.Option(None, "--provider", "-p",
                                 help="claude|anthropic|openai|ollama|external|lmstudio|llamacpp|vllm|jan|koboldcpp"),
    base_url: str = typer.Option(None, "--base-url",
                                 help="Server base URL (required for external; overrides any default)."),
    list_providers: bool = typer.Option(False, "--list",
                                        help="List discovered local LLM providers and exit."),
    config: str = typer.Option(None, "--config"),
):
    """Show or switch the generation backend/model in raggity.toml."""
    from pathlib import Path as _Path  # noqa: PLC0415
    from . import providers  # noqa: PLC0415

    if list_providers:
        _print_provider_table(providers.discover())
        return

    if model_name is None:
        gen = load_config(config).generation
        line = f"generation: backend={gen.backend} model={gen.model}"
        if gen.base_url:
            line += f" base_url={gen.base_url}"
        console.print(line)
        return

    if provider is not None and provider not in providers.BACKEND_ALIASES:
        console.print(
            f"[red]Invalid provider {provider!r}.[/red] Choices: "
            + ", ".join(providers.BACKEND_ALIASES)
        )
        raise typer.Exit(1)

    import tomlkit  # noqa: PLC0415
    target = _Path(config) if config else (_find_config_path(None) or _Path.cwd() / "raggity.toml")
    created = False
    if not target.exists():
        target.write_text(_INIT_TEMPLATE, encoding="utf-8")
        created = True

    doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    gen = doc.get("generation")
    if gen is None:
        gen = tomlkit.table()
        doc["generation"] = gen
    gen["model"] = model_name
    if provider is not None:
        backend, alias_base = providers.BACKEND_ALIASES[provider]
        gen["backend"] = backend
        if alias_base is not None:
            gen["base_url"] = alias_base
    if base_url is not None:
        gen["base_url"] = base_url

    # backend=external is externally managed (Rigma) and has no default base_url:
    # require one before persisting so we never write an unusable config.
    if gen.get("backend") == "external" and not gen.get("base_url"):
        if created:
            target.unlink()  # don't leave a half-created template behind
        console.print(
            "[red]backend=external requires a base_url.[/red] "
            "Pass [cyan]--base-url <url>[/cyan] "
            "(e.g. http://127.0.0.1:9999) or set generation.base_url in the config."
        )
        raise typer.Exit(1)

    target.write_text(tomlkit.dumps(doc), encoding="utf-8")

    if created:
        console.print(f"[green]Created[/green] {target.name}")
    eff_backend = gen.get("backend", "claude")
    eff_model = gen.get("model")
    eff_base = gen.get("base_url")
    line = f"generation: backend={eff_backend} model={eff_model}"
    if eff_base:
        line += f" base_url={eff_base}"
    console.print(f"[green]{line}[/green]")
    if eff_backend == "ollama":
        console.print(
            "[dim]ollama base_url defaults to http://localhost:11434/v1; "
            f"pull the model first: ollama pull {eff_model}[/dim]"
        )
    elif eff_backend == "external":
        console.print(
            "[dim]external server is managed outside raggity (e.g. Rigma); "
            "raggity never starts it. Ensure it is running (rag doctor to check).[/dim]"
        )
    elif provider in ("lmstudio", "llamacpp", "vllm", "jan", "koboldcpp"):
        console.print(
            f"[dim]ensure the {provider} server is running (rag doctor to check).[/dim]"
        )


@app.command()
def doctor(config: str = typer.Option(None, "--config")):
    """Run environment diagnostics (config, index, embedding, generation backend)."""
    from . import doctor as _doctor  # noqa: PLC0415
    raise typer.Exit(_doctor.run_doctor(config, console))


@app.command()
def ingest(config: str = typer.Option(None, "--config")):
    """Incrementally index configured source folders."""
    _check_no_config(config)
    report = _rag(config).ingest()
    if report.scanned > 10_000:
        console.print(
            f"[yellow]Matched {report.scanned} files - check your "
            r"\[sources] include/exclude patterns if unintended.[/yellow]"
        )
    console.print(
        f"[green]Indexed.[/green] added={report.added} updated={report.updated} "
        f"deleted={report.deleted} unchanged={report.unchanged} "
        f"skipped={report.skipped_generic}"
    )
    # Print install hints for any file types that need optional extras
    for extra, cnt in report.skipped_needs_extra.items():
        console.print(
            rf"[yellow]Skipped {cnt} file(s) needing raggity\[{extra}] - "
            rf"install with:[/yellow] [cyan]pip install raggity\[{extra}][/cyan]"
        )


@app.command(name="ingest-url")
def ingest_url(
    url: str = typer.Argument(..., help="URL to fetch and index."),
    depth: int = typer.Option(0, "--depth", "-d", help="BFS crawl depth (0 = start URL only)."),
    config: str = typer.Option(None, "--config"),
):
    """Fetch a URL (and optionally crawl same-domain links) and add to the index."""
    try:
        from .connectors.web import WebConnector  # noqa: PLC0415
    except ImportError:
        console.print(r"[red]ingest-url needs extra deps:[/red] pip install raggity\[web]")
        raise typer.Exit(1)
    rag = _rag(config)
    console.print(f"Fetching [cyan]{url}[/cyan] (depth={depth})...")
    try:
        connector = WebConnector(url, depth=depth, same_domain=True)
        docs = connector.fetch()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No content extracted.[/yellow]")
        raise typer.Exit(0)
    if depth == 0:
        # Single-page fetch: nothing in-scope can have vanished, and a raw-URL
        # prefix scope could wrongly prune other URLs sharing the string prefix.
        scope = None
    else:
        from urllib.parse import urlparse  # noqa: PLC0415
        pr = urlparse(url)
        # Trailing slash terminates the prefix so https://x.com never matches
        # https://x.company.com.
        scope = f"{pr.scheme}://{pr.netloc}/"
    added = rag.ingest_documents(docs, scope=scope)
    console.print(f"[green]Ingested {added} page(s) from {url}.[/green]")


@app.command(name="ingest-repo")
def ingest_repo(
    url: str = typer.Argument(..., help="Git repository URL to clone and index."),
    ref: str = typer.Option(None, "--ref", "-r", help="Branch or tag to check out (default: HEAD)."),
    config: str = typer.Option(None, "--config"),
):
    """Shallow-clone a git repository and add all text files to the index."""
    from .connectors.github import GitHubConnector  # noqa: PLC0415
    rag = _rag(config)
    console.print(f"Cloning [cyan]{url}[/cyan]" + (f" @ {ref}" if ref else "") + "...")
    try:
        connector = GitHubConnector(url, ref=ref)
        docs = connector.fetch()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No text files found in repository.[/yellow]")
        raise typer.Exit(0)
    added = rag.ingest_documents(docs, scope=f"{url}#")
    console.print(f"[green]Ingested {added} file(s) from {url}.[/green]")


@app.command(name="ingest-obsidian")
def ingest_obsidian(
    vault: str = typer.Argument(..., help="Path to the Obsidian vault directory."),
    config: str = typer.Option(None, "--config"),
):
    """Read all Markdown notes from an Obsidian vault and add them to the index."""
    from .connectors.obsidian import ObsidianConnector  # noqa: PLC0415
    rag = _rag(config)
    console.print(f"Reading vault [cyan]{vault}[/cyan]...")
    try:
        connector = ObsidianConnector(vault)
        docs = connector.fetch()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not docs:
        console.print("[yellow]No Markdown notes found in vault.[/yellow]")
        raise typer.Exit(0)
    from pathlib import Path as _Path  # noqa: PLC0415
    scope = _Path(vault).as_posix().rstrip("/") + "/"
    added = rag.ingest_documents(docs, scope=scope)
    console.print(f"[green]Ingested {added} note(s) from {vault}.[/green]")


@app.command(name="graph-build")
def graph_build(config: str = typer.Option(None, "--config")):
    """Extract entities/relations from all indexed chunks and build graph.json.

    Requires retrieval.graph = true in the config. LLM-cost-heavy: one provider
    call per indexed chunk. Run after `rag ingest` to populate the graph, or use
    `rag ingest` with retrieval.graph = true to do both in one step.
    """
    import asyncio
    rag = _rag(config)
    if not rag.cfg.retrieval.graph:
        console.print("[red]Error:[/red] retrieval.graph must be true to run graph-build.")
        raise typer.Exit(1)
    n = rag.store.count()
    if n == 0:
        console.print("[yellow]Index is empty - run `rag ingest` first.[/yellow]")
        raise typer.Exit(1)
    console.print(f"Building graph from [cyan]{n}[/cyan] chunks (+{n} LLM calls)...")
    try:
        asyncio.run(rag.build_graph())
    except Exception as exc:
        console.print(f"[red]Graph build failed:[/red] {exc}")
        raise typer.Exit(1)
    console.print("[green]Graph built.[/green]")


@app.command()
def ask(question: str, config: str = typer.Option(None, "--config"),
        plain: bool = typer.Option(False, "--plain"),
        expand: bool = typer.Option(False, "--expand"),
        hyde: bool = typer.Option(False, "--hyde"),
        step_back: bool = typer.Option(False, "--step-back"),
        no_stream: bool = typer.Option(False, "--no-stream"),
        decompose: bool = typer.Option(False, "--decompose"),
        no_cache: bool = typer.Option(False, "--no-cache")):
    """Ask a question against your knowledge base."""
    import asyncio
    if _check_no_config(config):
        raise typer.Exit(0)
    rag = _rag(config)
    if rag.store.count() == 0:
        console.print(_EMPTY_KB_HINT)
        raise typer.Exit(0)
    if decompose:
        if expand or hyde or step_back:
            typer.echo("note: --decompose overrides other query transforms", err=True)
        typer.echo("Decomposing query (+model calls)...", err=True)
        answer = rag.ask_decompose(question)
        if plain:
            typer.echo(answer.text)
        else:
            console.print(answer.text)
    else:
        if expand or hyde or step_back:
            typer.echo("Query transforms enabled (+model calls)...", err=True)
        expand_arg = True if expand else None
        hyde_arg = True if hyde else None
        step_back_arg = True if step_back else None
        use_cache_arg = False if no_cache else None
        if plain or no_stream:
            # Buffered path - honors cache (unless --no-cache)
            answer = rag.ask(question, expand=expand_arg, hyde=hyde_arg, step_back=step_back_arg,
                             use_cache=use_cache_arg)
            if plain:
                typer.echo(answer.text)
            else:
                console.print(answer.text)
        else:
            # Streaming path - default; always calls the model (cache is buffered-only)
            async def _stream():
                final = None
                async for piece in rag.aask_stream(question, expand=expand_arg,
                                                   hyde=hyde_arg, step_back=step_back_arg):
                    if isinstance(piece, str):
                        print(piece, end="", flush=True)
                    else:
                        final = piece
                print()
                return final
            answer = asyncio.run(_stream())
    if answer is not None and answer.citations and not plain:
        console.print("\n[dim]Sources:[/dim]")
        seen = set()
        for c in answer.citations:
            if c.supported and c.source_path not in seen:
                seen.add(c.source_path)
                console.print(f"  [dim]- {c.source_path}[/dim]")


@app.command()
def chat(config: str = typer.Option(None, "--config")):
    """Start an interactive multi-turn chat REPL against your knowledge base."""
    from .conversation import Conversation  # noqa: PLC0415
    if _check_no_config(config):
        raise typer.Exit(0)
    rag = _rag(config)
    if rag.store.count() == 0:
        console.print(_EMPTY_KB_HINT)
    conversation = Conversation()
    console.print("[green]raggity chat[/green] - type your question, 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            console.print("[dim]Bye![/dim]")
            break
        async def _stream_chat():
            # Build retrieval query from conversation history
            retrieval_q = conversation.retrieval_query(question)
            chunks = rag.retriever.retrieve(retrieval_q)
            history = conversation.recent(6) or None
            parts: list[str] = []
            final = None
            console.print("[cyan]raggity:[/cyan] ", end="")
            async for piece in rag.answerer.answer_stream(question, chunks, history=history):
                if isinstance(piece, str):
                    print(piece, end="", flush=True)
                    parts.append(piece)
                else:
                    final = piece
            print()
            conversation.add("user", question)
            conversation.add("assistant", "".join(parts).strip())
            return final

        answer = _run_async(_stream_chat())
        if answer is not None and answer.citations:
            seen: set[str] = set()
            footnotes = []
            for c in answer.citations:
                if c.supported and c.source_path not in seen:
                    seen.add(c.source_path)
                    footnotes.append(c.source_path)
            if footnotes:
                console.print("\n[dim]Sources: " + ", ".join(footnotes) + "[/dim]")
        console.print()


@app.command()
def status(config: str = typer.Option(None, "--config")):
    """Show index statistics."""
    rag = _rag(config)
    st = rag.status()
    for k, v in st.items():
        console.print(f"{k}: {v}")
    if rag.store.count() == 0:
        console.print(_EMPTY_KB_HINT)


@app.command()
def reindex(config: str = typer.Option(None, "--config"),
            force: bool = typer.Option(False, "--force")):
    """Rebuild the index from scratch."""
    cfg = load_config(config)
    if force:
        shutil.rmtree(cfg.index.path, ignore_errors=True)
    report = Raggity(cfg).ingest()
    console.print(f"[green]Reindexed.[/green] added={report.added}")


@app.command(name="eval")
def eval_cmd(golden: str = typer.Argument(...),
             config: str = typer.Option(None, "--config"),
             k: int = typer.Option(5, "--k"),
             llm_judge: bool = typer.Option(False, "--llm-judge")):
    """Run free CPU retrieval metrics against a golden.jsonl set."""
    rag = _rag(config)
    if llm_judge:
        typer.echo("Running LLM-judge eval (+2 model calls per question)...", err=True)
        import asyncio
        from .evaluate import llm_judge as run_judge
        res = asyncio.run(run_judge(rag, load_golden(golden), rag.provider))
        console.print(f"Faithfulness={res.faithfulness:.3f}  "
                      f"AnswerRelevance={res.answer_relevance:.3f}  (n={res.n})")
        console.print("(note: self-assessed - same model family generates and grades)")
    else:
        res = evaluate(rag.retriever, load_golden(golden), k=k)
        console.print(f"Hit@{k}={res.hit_rate:.3f}  MRR={res.mrr:.3f}  "
                      f"Recall@{k}={res.recall:.3f}  (n={res.n})")


@app.command()
def watch(config: str = typer.Option(None, "--config"),
          debounce: float = typer.Option(2.0, "--debounce")):
    """Watch source folders and re-index on change (Ctrl-C to stop)."""
    rag = _rag(config)
    try:
        from .watch import run_watch
    except ImportError:
        console.print(r"[red]watch needs extra deps:[/red] pip install raggity\[watch]")
        raise typer.Exit(1)
    try:
        observer = run_watch(rag, rag.cfg.sources.include, debounce)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Watching[/green] {len(rag.cfg.sources.include)} source pattern(s). Ctrl-C to stop.")
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@app.command()
def serve(config: str = typer.Option(None, "--config"),
          host: str = typer.Option("127.0.0.1", "--host"),
          port: int = typer.Option(8000, "--port"),
          open: bool = typer.Option(False, "--open", help="Open the web UI in the default browser.")):
    """Run the local HTTP API server (with optional web UI)."""
    try:
        import uvicorn
        from .server import create_app
    except ImportError:
        console.print(r"[red]The server needs extra deps:[/red] pip install raggity\[server]")
        raise typer.Exit(1)
    if open:
        try:
            _open_browser_delayed(f"http://{host}:{port}", delay=1.2)
        except Exception:  # noqa: BLE001
            pass
    uvicorn.run(create_app(load_config(config)), host=host, port=port)


if __name__ == "__main__":
    app()
