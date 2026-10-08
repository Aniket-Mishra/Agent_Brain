from pathlib import Path

import pytest

from brain import index
from brain.notes import Note, chunk_note, read_note
from brain.search import Hit, Scope, by_words, load_hits


def test_frontmatter_offsets_survive_indexing_and_title_search(tmp_path: Path) -> None:
    path = tmp_path / "deadline.md"
    raw = "---\ntitle: Quasarunik\n---\n\n# Deadline\n\nDue 12 Oct.\n"
    path.write_text(raw, encoding="utf-8")
    note = read_note(path, tmp_path)
    assert note.body_start_line == 4

    connection = index.connect(tmp_path / "index.db")
    try:
        index.store_note(connection, note)
        hits = by_words(connection, "Quasarunik", 8, Scope())
        assert len(hits) == 1
        hit = hits[0]
        assert (hit.start_line, hit.end_line, hit.location) == (5, 7, "deadline.md:5-7")
        assert hit.text == "".join(raw.splitlines(keepends=True)[4:7])
        assert load_hits(connection, [hit.chunk_id], "read")[0].location == hit.location
    finally:
        connection.close()


def test_repeated_passages_keep_their_separate_source_lines() -> None:
    passage = "# Topic\n\nRepeated fact.\n\n"
    note = Note("repeat.md", "Repeat", passage * 2, [], "d", body_start_line=4)
    chunks = chunk_note(note, smallest=0)
    assert [chunk.text for chunk in chunks] == [passage, passage]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(4, 7), (8, 11)]


def test_merged_sections_keep_intervening_headings_and_original_spacing() -> None:
    body = "# Costs\n\nMonthly cost is $40.\n\n\n# Risk\n\nDowntime costs $500.\n"
    chunks = chunk_note(Note("costs.md", "Costs", body, [], "d"))
    assert len(chunks) == 1
    assert chunks[0].text == body
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 8)


@pytest.mark.parametrize("opening,closing", [("```python", "```"), ("~~~~python", "~~~~")])
def test_headings_inside_fences_stay_in_source_text(opening: str, closing: str) -> None:
    body = f"# Setup\n\nCommands:\n{opening}\n# Keep this command comment\nprint('ready')\n{closing}\n"
    chunks = chunk_note(Note("code.md", "Code", body, [], "d"), smallest=0)
    assert len(chunks) == 1
    assert chunks[0].heading == "Setup"
    assert chunks[0].text == body
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 7)


def test_shorter_fence_does_not_close_a_longer_fence() -> None:
    body = "````markdown\n```\n# A code example, not a section\n````\n# Notes\n\nDone.\n"
    chunks = chunk_note(Note("fence.md", "Fence", body, [], "d"), smallest=0)
    assert [chunk.heading for chunk in chunks] == ["", "Notes"]
    assert "".join(chunk.text for chunk in chunks) == body


def test_single_long_line_keeps_line_location_across_chunks() -> None:
    chunks = chunk_note(Note("long.md", "Long", "字" * 5000, [], "d", body_start_line=6))
    assert "".join(chunk.text for chunk in chunks) == "字" * 5000
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(6, 6)] * 3


@pytest.mark.parametrize("length,end_line", [(6, 11), (8, 12), (19, 13)])
def test_shortening_tracks_last_occupied_source_line(length: int, end_line: int) -> None:
    hit = Hit(1, "note.md", "Note", "", "First\nSecond\nThird\n", start_line=11, end_line=13)
    shortened = hit.shorten(length)
    assert shortened.text == hit.text[:length]
    assert shortened.end_line == end_line
    assert shortened.truncated == (length < len(hit.text))
    assert "[cut]" not in shortened.text
    assert shortened.excerpt == shortened.text + ("\n[cut]" if shortened.truncated else "")


def test_shortening_unicode_and_unknown_locations() -> None:
    hit = Hit(1, "note.md", "Note", "", "αβ\n第二\nfinal", start_line=4, end_line=6)
    shortened = hit.shorten(5)
    assert shortened.text == "αβ\n第二"
    assert shortened.location == "note.md:4-5"
    assert shortened.shorten(100).truncated
    assert hit.shorten(0).location == "note.md"
    assert Hit(2, "unknown.md", "Unknown", "", "Words").shorten(2).location == "unknown.md"
