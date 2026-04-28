from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from mnexa import ingest as ingest_mod
from mnexa import lint as lint_mod
from mnexa import query as query_mod
from mnexa import storage

# Load .env from cwd (or any parent dir) before any command runs.
load_dotenv()

app = typer.Typer(
    name="mnexa",
    help="A disciplined wiki maintainer for a personal markdown knowledge base.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def init(
    path: Path = typer.Argument(..., help="Where to create the new vault."),
) -> None:
    """Create a new vault at PATH."""
    storage.init_vault(path)


@app.command()
def ingest(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="File to ingest."),
) -> None:
    """Ingest a single source file into the wiki."""
    ingest_mod.run(file)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the wiki."),
) -> None:
    """Ask the wiki a question."""
    query_mod.run(question)


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Interactively fix lint findings."),
) -> None:
    """Audit the wiki for issues."""
    lint_mod.run(fix=fix)


if __name__ == "__main__":
    app()
