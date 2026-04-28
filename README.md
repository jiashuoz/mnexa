# Mnexa

A disciplined wiki maintainer for a personal markdown knowledge base. You drop documents into `raw/`; an LLM reads them and maintains a structured wiki of source / entity / concept pages with cross-references, an index, and a log. You curate; the LLM does the bookkeeping.

Implementation of the pattern in [Andrej Karpathy's LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — read that first; it's the design spec.

## Why

Most LLM document tools are RAG: retrieve chunks at query time, generate from the chunks, throw the synthesis away. Mnexa treats the wiki as a **persistent, compounding artifact** — every ingest updates entity and concept pages once, every query runs against accumulated synthesis instead of re-deriving from raw sources. Open the wiki in Obsidian, Logseq, VS Code, or any markdown editor. The LLM is the maintainer; you are the curator.

## Install

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv). Get a Gemini API key at <https://aistudio.google.com/apikey>.

```bash
git clone https://github.com/jiashuoz/mnexa
cd mnexa
uv sync
cp .env.example .env   # then paste your GOOGLE_API_KEY
```

## Use

```bash
# Create a new vault
uv run mnexa init ~/my-vault

# Drop a source into raw/, then ingest
cp some-paper.pdf ~/my-vault/raw/
cd ~/my-vault
uv run --project /path/to/mnexa mnexa ingest raw/some-paper.pdf

# Ask the wiki a question
uv run --project /path/to/mnexa mnexa query "what does this paper claim?"

# Audit the wiki
uv run --project /path/to/mnexa mnexa lint
```

## Vault layout

```
my-vault/
├── .git/
├── .gitignore                  # ignores .mnexa/ and .env
├── .mnexa/                     # Mnexa local state (lint reports)
├── CLAUDE.md                   # the schema — edit §6 to customize
├── raw/                        # immutable source documents
└── wiki/
    ├── index.md                # categorized table of contents
    ├── log.md                  # append-only activity log
    ├── sources/                # one page per ingested document
    ├── entities/               # people, orgs, products, places
    └── concepts/               # ideas, techniques, recurring topics
```

Every successful ingest is a git commit. Free undo, free history, free diff.

## How it works

**Ingest** is a two-stage pipeline:

1. **Analyze** — LLM reads the source plus the schema, index, and obviously-related existing pages. Produces a structured analysis (entities, concepts, claims, contradictions). Internal scratch.
2. **Generate** — LLM emits FILE blocks for the new/updated wiki pages. Mnexa parses, validates paths and frontmatter, **substring-verifies that every `⟦"..."⟧` source-quote marker appears verbatim in the source**, then atomically writes and commits.

The substring verifier is the anti-hallucination floor. If the LLM invents a biographical detail not present in the source, the marker check fails and the ingest aborts with no on-disk changes.

**Query** is a single LLM call against `index.md` + the top-N pages by keyword overlap, streamed to stdout with inline `[[wikilink]]` citations. Logged to `log.md`.

**Lint** runs deterministic checks first (broken links, frontmatter, index/wiki sync, orphans, ungrounded pages, slug style), then one LLM call for semantic checks (contradictions, stale claims, missing pages, slug typos). Output: `.mnexa/lint/<timestamp>.md`.

## LLM

Provider-agnostic via a small `LLMClient` protocol. v0 ships Google Gemini (default `gemini-3-flash-preview`). Set `MNEXA_MODEL` to any `gemini-*` model; set `MNEXA_PROVIDER` to override the auto-inference. Adding Anthropic or OpenAI is ~80 lines plus an extras entry — not shipped because no one needs it yet.

## Status

| | |
|---|---|
| `mnexa init` | ✅ |
| `mnexa ingest` | ✅ (`.md`, `.txt`, `.pdf`, `.docx`) |
| `mnexa query` | ✅ |
| `mnexa lint` | ✅ |
| `mnexa lint --fix` | not yet (v0.1) |
| save query answer as wiki page | not yet (v0.1) |
| Anthropic / OpenAI providers | not yet |

## Develop

```bash
uv sync --all-extras
uv run pytest         # 45 tests
uv run ruff check .
uv run pyright        # strict
```

Prompts live as files in [`src/mnexa/prompts/`](src/mnexa/prompts) and load via `importlib.resources`. Edit them, rerun, iterate.

## Design notes

- **Pure markdown is the canonical store.** No SQLite, no vector index, no FTS5. Karpathy's gist argues `index.md` is enough at moderate scale; we believe it until measurements say otherwise.
- **Two-stage ingest** is borrowed from [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki); the **deterministic-then-LLM lint tier** is borrowed from [SamurAIGPT/llm-wiki-agent](https://github.com/SamurAIGPT/llm-wiki-agent). The substring-grounding verifier is novel — neither reference project does it.
- Atomic-ish writes via stage-then-rename + `git checkout HEAD --` rollback on failure. The git commit is the durability barrier.
- Gemini context caching is a no-op at our schema size (~3k tokens, below the threshold). The protocol still expresses intent so other providers can honor it.
