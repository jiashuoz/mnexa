"""Two-stage ingest pipeline: analyze (Stage 1) → generate (Stage 2)."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import typer

from mnexa import storage
from mnexa.llm import LLMClient, Usage, get_client
from mnexa.parser import parse_file_blocks, verify_grounding
from mnexa.parsers import read_source
from mnexa.prompts import load as load_prompt

MAX_SOURCE_BYTES = 200_000
MAX_RELATED_PAGES = 10


def run(file: Path, *, client: LLMClient | None = None) -> None:
    asyncio.run(_run_async(file, client=client))


async def _run_async(file: Path, *, client: LLMClient | None) -> None:
    vault = storage.find_vault(Path.cwd())
    if vault is None:
        typer.echo(
            "error: not inside an Mnexa vault (run `mnexa init` first)", err=True
        )
        raise typer.Exit(1)

    file = file.expanduser().resolve()
    if not file.is_file():
        typer.echo(f"error: not a file: {file}", err=True)
        raise typer.Exit(1)

    source_bytes = file.read_bytes()
    if len(source_bytes) > MAX_SOURCE_BYTES:
        typer.echo(
            f"error: source is {len(source_bytes)} bytes; v0 limit is "
            f"{MAX_SOURCE_BYTES}. Split it into smaller files.",
            err=True,
        )
        raise typer.Exit(1)

    source_text = read_source(file)
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    raw_dest = vault / "raw" / file.name
    if file.parent.resolve() != (vault / "raw").resolve() and not raw_dest.exists():
        shutil.copy2(file, raw_dest)

    if client is None:
        client = get_client()

    schema = (vault / "CLAUDE.md").read_text(encoding="utf-8")
    index = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    related = _find_related_pages(vault, source_text, MAX_RELATED_PAGES)
    today = date.today().isoformat()

    typer.echo(f"[stage 1] analyzing {file.name}…", err=True)
    stage1_system = _build_system("stage1.md", schema)
    stage1_user = _build_stage1_user(
        index=index, related=related, vault=vault,
        filename=file.name, source_text=source_text,
    )
    completion = await client.complete(
        system=stage1_system, user=stage1_user, cache_system=True
    )
    analysis = completion.text
    typer.echo(f"[stage 1] done · {_fmt_usage(completion.usage)}", err=True)

    typer.echo("[stage 2] generating wiki updates…", err=True)
    existing = _gather_existing_pages(vault, analysis)
    stage2_system = _build_system("stage2.md", schema)
    stage2_user = _build_stage2_user(
        analysis=analysis, vault=vault, filename=file.name,
        source_text=source_text, source_hash=source_hash,
        existing=existing, today=today,
    )

    accumulated: list[str] = []
    async for chunk in client.stream(
        system=stage2_system, user=stage2_user, cache_system=True
    ):
        sys.stderr.write(chunk)
        sys.stderr.flush()
        accumulated.append(chunk)
    sys.stderr.write("\n")
    output = "".join(accumulated)
    if client.last_usage is not None:
        typer.echo(f"[stage 2] done · {_fmt_usage(client.last_usage)}", err=True)

    blocks = parse_file_blocks(output, vault)
    if not blocks:
        typer.echo("no changes (Stage 2 emitted no FILE blocks)", err=True)
        return

    verify_grounding(blocks, source_text)

    pages = {b.abs_path: b.raw_content for b in blocks}
    try:
        storage.write_pages(vault, pages)
    except Exception:
        storage.git_rollback(vault)
        raise

    if not storage.git_commit(vault, f"ingest: {file.name}"):
        typer.echo("warning: write succeeded but no git changes detected", err=True)
        return

    typer.echo(f"ingested {file.name} → {len(blocks)} pages updated", err=True)


_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "has", "have", "her", "his", "its", "may", "one", "our", "out",
    "she", "two", "way", "who", "with", "this", "that", "from", "they",
    "them", "their", "there", "what", "when", "where", "which", "while",
    "would", "could", "should", "than", "then", "into", "your",
})


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if len(t) > 2 and t not in _STOPWORDS}


def _find_related_pages(vault: Path, source_text: str, top_n: int) -> list[Path]:
    src_tokens = _tokens(source_text)
    if not src_tokens:
        return []
    wiki = vault / "wiki"
    scored: list[tuple[int, Path]] = []
    for p in wiki.rglob("*.md"):
        if p.name in {"index.md", "log.md"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        score = len(src_tokens & _tokens(text))
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_n]]


def _gather_existing_pages(vault: Path, analysis: str) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for m in re.finditer(r"wiki/[\w/\-.]+\.md", analysis):
        p = (vault / m.group(0)).resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            paths.append(p)
    for name in ("index.md", "log.md"):
        p = (vault / "wiki" / name).resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            paths.append(p)
    return paths


def _build_system(prompt_name: str, schema: str) -> str:
    return f"{load_prompt(prompt_name)}\n\n<schema>\n{schema}\n</schema>"


def _read_pages(paths: Iterable[Path], vault: Path) -> str:
    parts: list[str] = []
    for p in paths:
        rel = p.relative_to(vault)
        parts.append(f"--- {rel} ---\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def _build_stage1_user(
    *, index: str, related: list[Path], vault: Path, filename: str, source_text: str
) -> str:
    related_block = _read_pages(related, vault) if related else "(none)"
    return (
        f"<index>\n{index}\n</index>\n\n"
        f"<related_pages>\n{related_block}\n</related_pages>\n\n"
        f'<source filename="{filename}">\n{source_text}\n</source>'
    )


def _build_stage2_user(
    *, analysis: str, vault: Path, filename: str, source_text: str,
    source_hash: str, existing: list[Path], today: str,
) -> str:
    existing_block = _read_pages(existing, vault) if existing else "(none)"
    return (
        f"<analysis>\n{analysis}\n</analysis>\n\n"
        f'<source filename="{filename}" hash="{source_hash}">\n'
        f"{source_text}\n</source>\n\n"
        f"<existing_pages>\n{existing_block}\n</existing_pages>\n\n"
        f"<today>{today}</today>"
    )


def _fmt_usage(u: Usage) -> str:
    return f"in={u.input_tokens} out={u.output_tokens} cached={u.cached_input_tokens}"
