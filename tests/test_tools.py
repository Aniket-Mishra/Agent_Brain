from pathlib import Path

import pytest

from brain import index, ollama, tools
from brain.notes import Note
from brain.search import Scope


@pytest.fixture
def indexed_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notes = [
        Note("Focus.md", "Storage plan", "We chose SQLite for local storage. [[Linked]] [[Missing]]", ["Linked", "Missing"], "focus", 300, "research", created="2026-03-12"),
        Note("Peer.md", "Related storage", "SQLite stores the passages on this machine. [[Missing]]", ["Missing"], "peer", 300, "research", created="2026-03-14"),
        Note("Linked.md", "Linked storage", "SQLite also holds indexed link edges. [[Focus]]", ["Focus"], "linked", 300, "research", created="2026-03-16"),
        Note("Hidden.md", "Daily storage", "SQLite belongs to the daily notes. [[Connected]] [[Missing]]", ["Connected", "Missing"], "hidden", 300, "daily", created="2026-03-17"),
        Note("Connected.md", "Connected elsewhere", "This research note has an incoming daily link.", [], "connected", 300, "research", created="2026-03-18"),
        Note("Stub.md", "Unfinished", "An unfinished note with enough words to index.", [], "stub", 80, "research", created="2026-03-19"),
        Note("Undated.md", "No date", "SQLite also appears in this undated research note.", [], "undated", 300, "research"),
    ]
    connection = index.connect(tmp_path / "tools.db")
    for note in notes:
        index.store_note(connection, note)
        vector = [1.0, 0.0] if note.path != "Peer.md" else [0.8, 0.6]
        connection.execute("UPDATE chunks SET vector = ? WHERE path = ?", (index.pack(vector), note.path))
    index.rewrite_links(connection, notes)
    connection.commit()
    monkeypatch.setattr(ollama, "embed_query", lambda query, model: [1.0, 0.0])
    yield connection
    connection.close()


MARCH_RESEARCH = Scope(kinds=("research",), since="2026-03-01", until="2026-03-31")


def test_search_returns_scoped_evidence_with_metadata(indexed_tools) -> None:
    result = tools.search(indexed_tools, "SQLite", "stub-model", scope=MARCH_RESEARCH)
    assert result.evidence
    assert all(hit.kind == "research" and hit.created.startswith("2026-03") for hit in result.evidence)
    assert {hit.path for hit in result.evidence}.isdisjoint({"Hidden.md", "Undated.md"})
    assert not result.paths
    assert "SQLite" in tools.format_result(result)
    assert "Title: Storage plan" in tools.format_result(result)


@pytest.mark.parametrize("query", ["", " \n ", 42, None, "é" * 4097])
def test_invalid_queries_fail_before_embedding(indexed_tools, monkeypatch, query) -> None:
    def unexpected(*_args):
        pytest.fail("invalid query reached the embedding model")

    monkeypatch.setattr(ollama, "embed_query", unexpected)
    with pytest.raises(ValueError, match="query must be nonempty"):
        tools.search(indexed_tools, query, "stub-model")


@pytest.mark.parametrize("limit", [0, 26, True, 1.5])
def test_search_rejects_unbounded_or_malformed_limits(indexed_tools, limit) -> None:
    with pytest.raises(ValueError, match="limit must be an integer"):
        tools.search(indexed_tools, "SQLite", "stub-model", limit=limit)


@pytest.mark.parametrize("tool", [tools.read, tools.backlinks, tools.related])
@pytest.mark.parametrize("path", ["/Focus.md", "C:/Focus.md", "C:Focus.md", "../Focus.md", "a/../Focus.md", "a//Focus.md", "./Focus.md", "a\\Focus.md", "", "é" * 2049])
def test_path_tools_reject_malformed_paths(indexed_tools, tool, path) -> None:
    result = tool(indexed_tools, path)
    assert result.text == f"unreadable: {path}"
    assert not result.evidence and not result.paths


@pytest.mark.parametrize("tool", [tools.read, tools.backlinks, tools.related])
def test_path_tools_reject_missing_and_out_of_scope_notes(indexed_tools, tool) -> None:
    assert tool(indexed_tools, "Absent.md").text == "unreadable: Absent.md"
    assert tool(indexed_tools, "Hidden.md", MARCH_RESEARCH).text == "unreadable: Hidden.md"


def test_read_uses_indexed_passages_and_preserves_heading(indexed_tools) -> None:
    connection = indexed_tools
    connection.execute("UPDATE chunks SET heading = 'Storage > Decision' WHERE path = 'Focus.md'")
    result = tools.read(connection, "Focus.md", MARCH_RESEARCH)
    assert len(result.evidence) == 1
    hit = result.evidence[0]
    assert (hit.path, hit.title, hit.heading) == ("Focus.md", "Storage plan", "Storage > Decision")
    assert "We chose SQLite" in hit.text
    assert not result.paths
    output = tools.format_result(result)
    assert output.startswith("Indexed passages")
    assert "Heading: Storage > Decision" in output


def test_read_truncates_passages_without_losing_metadata(indexed_tools) -> None:
    connection = indexed_tools
    connection.execute("DELETE FROM chunks WHERE path = 'Focus.md'")
    connection.executemany(
        "INSERT INTO chunks(path, heading, text) VALUES ('Focus.md', ?, ?)",
        [("First", "x" * 4000), ("Second", "y" * 4000), ("Last", "z" * 500)],
    )
    result = tools.read(connection, "Focus.md")
    assert sum(len(hit.text) for hit in result.evidence) == 6000
    assert [hit.heading for hit in result.evidence] == ["First", "Second"]
    assert result.evidence[-1].text.endswith("[truncated]")
    assert all(hit.title == "Storage plan" for hit in result.evidence)


def test_read_distinguishes_empty_indexed_note_from_missing(indexed_tools) -> None:
    connection = indexed_tools
    connection.execute("DELETE FROM chunks WHERE path = 'Stub.md'")
    result = tools.read(connection, "Stub.md")
    assert result.text == "No indexed passages in Stub.md."
    assert not result.evidence


def test_backlinks_filter_and_cap_distinct_notes(indexed_tools) -> None:
    connection = indexed_tools
    for number in range(30):
        path = f"Research {number:02d}.md"
        connection.execute("INSERT INTO notes(path, title, digest, kind, created) VALUES (?, ?, '', 'research', '2026-03-10')", (path, path))
        connection.executemany("INSERT INTO links(source, target, raw) VALUES (?, 'Focus.md', ?)", [(path, "Focus"), (path, "Focus.md")])
    connection.execute("INSERT INTO links(source, target, raw) VALUES ('Hidden.md', 'Focus.md', 'Focus')")
    result = tools.backlinks(connection, "Focus.md", MARCH_RESEARCH)
    assert len(result.paths) == 25
    assert "Hidden.md" not in result.paths
    assert result.text.endswith("and 6 more")
    assert tools.format_result(result) == result.text


def test_related_filters_before_limiting_and_skips_linked_notes(indexed_tools) -> None:
    connection = indexed_tools
    for number in range(8):
        path = f"Daily {number}.md"
        connection.execute("INSERT INTO notes(path, title, digest, kind, created) VALUES (?, ?, '', 'daily', '2026-03-10')", (path, path))
        connection.execute("INSERT INTO chunks(path, text, vector) VALUES (?, 'Daily storage', ?)", (path, index.pack([1.0, 0.0])))
    result = tools.related(connection, "Focus.md", MARCH_RESEARCH)
    assert "Peer.md" in result.paths
    assert "Linked.md" not in result.paths
    assert "Hidden.md" not in result.paths
    assert "Undated.md" not in result.paths
    assert len(result.paths) <= 5
    assert not result.evidence


def test_survey_uses_scoped_counts_and_global_link_classification(indexed_tools, monkeypatch) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("survey called a model")

    monkeypatch.setattr(ollama, "embed_query", unexpected)
    result = tools.survey(indexed_tools, MARCH_RESEARCH)
    assert "Notes: 5\nPassages: 5\nLinks from scoped notes: 4" in result.text
    assert "Kinds: research (5)" in result.text
    assert "Missing (2 notes)" in result.text
    assert result.paths == ("Stub.md", "Peer.md")
    assert "Connected.md" not in result.paths
    assert "Hidden.md" not in result.text
    assert not result.evidence


def test_survey_empty_scope_is_explicit(indexed_tools) -> None:
    result = tools.survey(indexed_tools, Scope(kinds=("absent",)))
    assert "Notes: 0\nPassages: 0\nLinks from scoped notes: 0" in result.text
    assert "Kinds: none" in result.text
    assert not result.paths


def test_survey_reports_only_the_displayed_paths(indexed_tools) -> None:
    connection = indexed_tools
    connection.executemany(
        "INSERT INTO notes(path, title, digest, kind) VALUES (?, '', '', 'extra')",
        [(f"Extra {number:02d}.md",) for number in range(20)],
    )
    result = tools.survey(connection, Scope(kinds=("extra",)))
    assert result.paths == tuple(f"Extra {number:02d}.md" for number in range(10))
    assert result.text.count("and 10 more") == 2
    assert "Extra 10.md" not in result.text
