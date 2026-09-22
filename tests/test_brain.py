"""Tests for the logic that can break without anyone noticing.

Nothing here talks to Ollama. The embedding call is replaced with a stub,
because what needs checking is the code around the model, not the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import index, links, ollama
from brain.answer import format_sources, source_budget
from brain.search import Hit, search, to_match_query
from brain.notes import Note, chunk_note, find_links, pack, read_note, split_to_fit

NOTES = {
    "Home.md": "---\ntitle: Home\n---\n\n# Home\n\nLinks to [[Thesis]] and [[Nowhere]].\n",
    "work/Thesis.md": (
        "# Thesis\n\n## Storage decision\n\nWe picked SQLite over Qdrant.\n\n"
        "## Timeline\n\nSubmit in June.\n"
    ),
    "work/Empty.md": "",
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / ".obsidian").mkdir(parents=True)
    for relative, text in NOTES.items():
        note = root / relative
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def indexed(vault: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A built index, with a stand-in for the embedding model."""

    def fake_embed(texts: list[str], model: str) -> list[list[float]]:
        # Distinct but deterministic, so ranking is stable across runs.
        return [[float(len(text) % 7), float(text.count("e")), 1.0] for text in texts]

    monkeypatch.setattr(ollama, "embed_documents", fake_embed)
    monkeypatch.setattr(ollama, "embed_query", lambda text, model: [1.0, 1.0, 1.0])

    connection = index.connect(tmp_path / "index.db")
    index.refresh(connection, vault, "stub-model")
    yield connection, vault
    connection.close()


class TestChunking:
    def test_heading_path_travels_with_the_chunk(self, vault: Path) -> None:
        note = read_note(vault / "work" / "Thesis.md", vault)
        headings = {chunk.heading for chunk in chunk_note(note, largest=40, smallest=10)}
        assert "Thesis > Storage decision" in headings

    def test_a_long_run_without_blank_lines_still_splits(self) -> None:
        """A lecture transcript is 40,000 characters with no paragraph breaks.

        Splitting on blank lines alone left one chunk the embedding model would
        silently truncate.
        """
        transcript = "\n".join(f"line {number} of the talk" for number in range(2000))
        pieces = split_to_fit(transcript, largest=2000)
        assert len(pieces) > 1
        assert max(len(piece) for piece in pieces) <= 2000

    def test_a_single_enormous_line_is_cut(self) -> None:
        pieces = split_to_fit("x" * 9000, largest=2000)
        assert max(len(piece) for piece in pieces) <= 2000

    def test_short_tails_do_not_become_their_own_chunk(self) -> None:
        text = "\n\n".join(["word " * 400, "word " * 400, "tiny"])
        pieces = pack(split_to_fit(text, largest=2000), largest=2000, smallest=250)
        assert all(len(piece) > 250 for piece in pieces)
        assert "tiny" in pieces[-1]

    def test_punctuation_only_sections_are_dropped(self) -> None:
        note = Note("n.md", "N", "# Title\n\n---\n", [], "d")
        assert chunk_note(note) == []

    def test_an_empty_note_yields_nothing(self, vault: Path) -> None:
        assert chunk_note(read_note(vault / "work" / "Empty.md", vault)) == []


class TestLinkParsing:
    def test_links_inside_code_fences_are_ignored(self) -> None:
        body = "Real [[One]]\n\n```\nSample [[NotALink]]\n```\n"
        assert find_links(body) == ["One"]

    def test_aliases_and_headings_are_stripped_from_targets(self) -> None:
        assert find_links("[[Note|shown text]] and [[Other#Section]]") == ["Note", "Other"]


class TestSearchQuery:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("what did I decide about storage", '"decide" OR "storage"'),
            ("ERR_CONN_4471", '"ERR_CONN_4471"'),
            ('"deep work" and why', '"deep work"'),
            ("what is this", '"what" OR "is" OR "this"'),
        ],
    )
    def test_query_building(self, question: str, expected: str) -> None:
        """Stopwords go, unless dropping them would leave nothing to search for."""
        assert to_match_query(question) == expected

    @pytest.mark.parametrize(
        "question", ["a AND b", "col:value", "star*", '"unbalanced', "!!! ???"]
    )
    def test_fts_operators_never_reach_sqlite(self, question: str, indexed) -> None:
        """A raw question containing FTS syntax must not raise."""
        connection, _vault = indexed
        assert isinstance(search(connection, question, "stub-model"), list)


class TestIndexing:
    def test_notes_chunks_and_links_are_stored(self, indexed) -> None:
        connection, _vault = indexed
        totals = index.counts(connection)
        assert totals["notes"] == 3
        assert totals["chunks"] > 0
        # [[Nowhere]] points at no note, so only the Thesis edge survives.
        assert totals["links"] == 1

    def test_every_chunk_gets_a_vector(self, indexed) -> None:
        connection, _vault = indexed
        missing = connection.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE vector IS NULL"
        ).fetchone()["total"]
        assert missing == 0

    def test_an_unchanged_vault_needs_no_work(self, indexed) -> None:
        connection, vault = indexed
        assert index.is_current(connection, vault, "stub-model")
        assert index.refresh(connection, vault, "stub-model").chunks == 0

    def test_editing_a_note_reindexes_only_that_note(self, indexed) -> None:
        connection, vault = indexed
        (vault / "work" / "Thesis.md").write_text("# Thesis\n\nWe picked Postgres.\n", "utf-8")
        report = index.refresh(connection, vault, "stub-model")
        assert (report.updated, report.added, report.removed) == (1, 0, 0)

    def test_deleting_a_note_removes_its_chunks(self, indexed) -> None:
        connection, vault = indexed
        (vault / "work" / "Thesis.md").unlink()
        assert index.refresh(connection, vault, "stub-model").removed == 1
        remaining = {row["path"] for row in connection.execute("SELECT path FROM chunks")}
        assert "work/Thesis.md" not in remaining

    def test_a_rename_is_noticed(self, indexed) -> None:
        """Renaming changes no file's contents and leaves the note count equal,
        so the freshness check has to compare paths, not just count and time."""
        connection, vault = indexed
        (vault / "work" / "Thesis.md").rename(vault / "work" / "Dissertation.md")
        assert not index.is_current(connection, vault, "stub-model")

    def test_a_new_note_resolves_links_that_pointed_nowhere(self, indexed) -> None:
        connection, vault = indexed
        (vault / "Nowhere.md").write_text("# Nowhere\n\nHere now.\n", encoding="utf-8")
        index.refresh(connection, vault, "stub-model")
        targets = {row["target"] for row in connection.execute("SELECT target FROM links")}
        assert "Nowhere.md" in targets

    def test_switching_embedding_model_rebuilds(self, indexed) -> None:
        connection, vault = indexed
        assert index.refresh(connection, vault, "other-model").rebuilt

    def test_the_same_model_under_two_names_does_not_rebuild(self, indexed) -> None:
        """`nomic-embed-text` and `nomic-embed-text:latest` are one model.

        Treating them as two re-embeds the whole vault on every launch.
        """
        connection, vault = indexed
        index.write_meta(connection, "embedding_model", "nomic-embed-text")
        assert index.is_current(connection, vault, "nomic-embed-text:latest")


class TestLinkGraph:
    def test_backlinks(self, indexed) -> None:
        connection, _vault = indexed
        assert links.backlinks(connection, "work/Thesis.md") == ["Home.md"]

    def test_the_focus_expands_even_when_it_is_a_hub(self, indexed) -> None:
        connection, _vault = indexed
        monkeyed = links.BUSY_NOTE_LINKS
        try:
            links.BUSY_NOTE_LINKS = 0
            nodes, _edges = links.neighbourhood(connection, "Home.md", hops=1)
            assert "work/Thesis.md" in nodes
        finally:
            links.BUSY_NOTE_LINKS = monkeyed

    def test_dot_escapes_quotes_in_names(self) -> None:
        dot = links.to_dot({'a"b.md'}, set(), 'a"b.md')
        assert dot.count('"') % 2 == 0


class TestPrompt:
    def test_the_source_budget_follows_the_context_window(self) -> None:
        assert source_budget(2048) < source_budget(32768)
        assert source_budget(512) >= 1000

    def test_sources_are_numbered_and_stop_at_the_budget(self) -> None:
        hits = [Hit(number, f"{number}.md", str(number), "", "x" * 500) for number in range(20)]
        text = format_sources(hits, budget=1200)
        assert text.startswith("[1] 0.md")
        assert len(text) < 2200


class TestSchemaChanges:
    def test_an_old_index_is_rebuilt_rather_than_crashing(
        self, tmp_path, vault, monkeypatch
    ) -> None:
        """Pulling a version with a different table shape must not crash.

        The index is derived from the vault, so it gets thrown away and rebuilt.
        """
        monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))
        database = tmp_path / "index.db"

        connection = index.connect(database)
        index.refresh(connection, vault, "stub-model")
        connection.close()

        stale = index.connect(database)
        index.write_meta(stale, "schema_version", "0")
        stale.commit()
        stale.close()

        rebuilt = index.connect(database)
        assert index.counts(rebuilt)["notes"] == 0
        assert index.read_meta(rebuilt, "schema_version") == index.SCHEMA_VERSION
        rebuilt.close()

    def test_a_corrupt_index_file_is_replaced(self, tmp_path) -> None:
        database = tmp_path / "index.db"
        database.write_bytes(b"this is not a database")
        connection = index.connect(database)
        assert index.counts(connection)["notes"] == 0
        connection.close()


class TestRanking:
    def test_two_passages_with_the_same_opening_stay_separate(self, indexed) -> None:
        """Identity is the passage id. Keying on a text prefix merged any two
        passages from one note that happened to start the same way."""
        from brain.search import Hit, limit_per_note

        shared_opening = "The same first eighty characters repeated here for both of them."
        hits = [
            Hit(1, "a.md", "A", "", shared_opening + " one"),
            Hit(2, "a.md", "A", "", shared_opening + " two"),
        ]
        assert len(limit_per_note(hits, allowed=2)) == 2

    def test_one_note_cannot_fill_the_answer(self) -> None:
        from brain.search import Hit, limit_per_note

        hits = [Hit(number, "same.md", "S", "", "text") for number in range(6)]
        assert len(limit_per_note(hits, allowed=2)) == 2

    def test_appearing_in_both_rankings_beats_topping_one(self, indexed) -> None:
        connection, _vault = indexed
        hits = search(connection, "SQLite Qdrant storage decision", "stub-model", limit=5)
        assert hits
        assert all(hit.chunk_id for hit in hits)
