from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import typer

from mnexa import ingest as ingest_mod
from mnexa.ingest import classify_target


def test_classify_granola_note_url() -> None:
    t = classify_target("https://app.granola.ai/notes/abc12345")
    assert t.kind == "granola-note"
    assert t.external_id == "abc12345"


def test_classify_granola_note_scheme() -> None:
    t = classify_target("granola://note/abc12345")
    assert t.kind == "granola-note"
    assert t.external_id == "abc12345"


def test_classify_granola_list_bare() -> None:
    t = classify_target("granola")
    assert t.kind == "granola-list"
    assert t.since is None


def test_classify_granola_list_recent() -> None:
    t = classify_target("granola://recent")
    assert t.kind == "granola-list"


def test_classify_granola_list_with_since() -> None:
    t = classify_target("granola://since/2026-04-01")
    assert t.kind == "granola-list"
    assert t.since == "2026-04-01"


def test_classify_granola_unknown_form_errors() -> None:
    with pytest.raises(typer.BadParameter):
        classify_target("granola://wat/123")


def test_load_granola_source_renders_text() -> None:
    """Verify _load_granola_source flattens a note into IngestSource cleanly."""
    from mnexa.granola.client import GranolaNote

    note = GranolaNote(
        note_id="abc12345",
        title="Design review",
        summary="Decided to ship the feature next week.",
        transcript=[
            {"speaker": "Alice", "text": "Let's ship it."},
            {"speaker": "Bob", "text": "Agreed."},
        ],
        created_at="2026-04-15T14:00:00Z",
        modified_at="2026-04-15T15:30:00Z",
        owner_name="Alice Smith",
        owner_email="alice@example.com",
        participants=["Alice Smith", "Bob Jones"],
        raw={},
    )
    fake_client = MagicMock()
    fake_client.get_note.return_value = note

    source = ingest_mod._load_granola_source("abc12345", fake_client)  # pyright: ignore[reportPrivateUsage]

    assert source.filename == "Design review"
    assert "Decided to ship the feature next week." in source.text
    assert "Alice: Let's ship it." in source.text
    assert source.source_path == "granola://abc12345"
    assert source.granola_meta is not None
    assert source.granola_meta.note_id == "abc12345"
    assert source.granola_meta.participants == ["Alice Smith", "Bob Jones"]
    assert source.granola_meta.web_view_link == "https://app.granola.ai/notes/abc12345"


def test_render_note_text_handles_minimal_note() -> None:
    from mnexa.granola.client import GranolaNote, render_note_text

    note = GranolaNote(
        note_id="x",
        title="Quick sync",
        summary="",
        transcript=[],
        created_at="",
        modified_at="",
        owner_name=None,
        owner_email=None,
        participants=[],
        raw={},
    )
    text = render_note_text(note)
    assert "Quick sync" in text
