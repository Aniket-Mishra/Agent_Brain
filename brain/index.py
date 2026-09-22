"""The index: one SQLite file holding chunks, their vectors, and the link graph."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain import ollama
from brain.notes import Note, chunk_note, embedding_text, find_notes, read_note

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    path   TEXT PRIMARY KEY,
    title  TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    path    TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    heading TEXT NOT NULL DEFAULT '',
    text    TEXT NOT NULL,
    vector  BLOB
);
CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, content='chunks', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_after_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading)
    VALUES ('delete', old.id, old.text, old.heading);
END;

CREATE TABLE IF NOT EXISTS links (
    source TEXT NOT NULL,
    target TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS links_by_target ON links(target);
CREATE INDEX IF NOT EXISTS links_by_source ON links(source);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Report:
    added: int = 0
    updated: int = 0
    removed: int = 0
    chunks: int = 0
    seconds: float = 0.0
    rebuilt: bool = False


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() and read_schema_version(database) != SCHEMA_VERSION:
        discard(database)

    connection = sqlite3.connect(database, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA)
    write_meta(connection, "schema_version", SCHEMA_VERSION)
    connection.commit()
    return connection


def read_schema_version(database: Path) -> str | None:
    try:
        with sqlite3.connect(database) as probe:
            row = probe.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            return row[0] if row else None
    except sqlite3.DatabaseError:
        return None


def discard(database: Path) -> None:
    """Throw the index away when its layout is from an older version.

    Everything here is derived from the vault, so rebuilding costs a minute and
    migrating costs a migration system. Pulling a new version on another device
    would otherwise crash against the old table shape.
    """
    database.unlink(missing_ok=True)
    for companion in ("-wal", "-shm"):
        Path(str(database) + companion).unlink(missing_ok=True)


def read_meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def write_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def vault_fingerprint(vault: Path) -> str:
    """A cheap signature of the vault, from paths and modification times only.

    Streamlit re-runs the whole script on every interaction, so the freshness
    check has to be cheap enough to run constantly. This reads no file contents
    and still notices edits, additions, deletions and renames.
    """
    entries = sorted(
        (path.relative_to(vault).as_posix(), path.stat().st_mtime) for path in find_notes(vault)
    )
    return hashlib.blake2b(repr(entries).encode("utf-8"), digest_size=8).hexdigest()


def is_current(connection: sqlite3.Connection, vault: Path, model: str) -> bool:
    if not same_model(read_meta(connection, "embedding_model"), model):
        return False
    return read_meta(connection, "vault_fingerprint") == vault_fingerprint(vault)


def same_model(stored: str | None, wanted: str) -> bool:
    return stored is not None and ollama.canonical(stored) == ollama.canonical(wanted)


def refresh(
    connection: sqlite3.Connection,
    vault: Path,
    model: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Report:
    """Bring the index in line with what is on disk, re-embedding only what changed."""
    started = time.perf_counter()
    report = Report()
    announce = progress or (lambda *_: None)

    stored_model = read_meta(connection, "embedding_model")
    if stored_model is not None and not same_model(stored_model, model):
        # Vectors from two models are not comparable, so there is nothing to keep.
        connection.execute("DELETE FROM notes")
        report.rebuilt = True

    known = {row["path"]: row["digest"] for row in connection.execute("SELECT path, digest FROM notes")}
    announce("reading notes", 0, 1)
    notes = [read_note(path, vault) for path in find_notes(vault)]
    on_disk = {note.path: note for note in notes}

    gone = [path for path in known if path not in on_disk]
    if gone:
        connection.executemany("DELETE FROM notes WHERE path = ?", [(path,) for path in gone])
        report.removed = len(gone)

    changed = [note for note in notes if known.get(note.path) != note.digest]
    report.added = sum(1 for note in changed if note.path not in known)
    report.updated = len(changed) - report.added

    for position, note in enumerate(changed, start=1):
        announce("reading notes", position, len(changed))
        report.chunks += store_note(connection, note)

    embed_missing(connection, model, announce)
    rewrite_links(connection, notes)
    write_meta(connection, "embedding_model", model)
    write_meta(connection, "vault_fingerprint", vault_fingerprint(vault))
    write_meta(connection, "indexed_at", time.time())
    connection.commit()

    report.seconds = time.perf_counter() - started
    return report


def store_note(connection: sqlite3.Connection, note: Note) -> int:
    """Write the note and its passages, leaving the vectors for the batch pass."""
    connection.execute("DELETE FROM notes WHERE path = ?", (note.path,))
    connection.execute(
        "INSERT INTO notes(path, title, digest) VALUES (?, ?, ?)",
        (note.path, note.title, note.digest),
    )

    chunks = chunk_note(note)
    connection.executemany(
        "INSERT INTO chunks(path, heading, text) VALUES (?, ?, ?)",
        [(chunk.path, chunk.heading, chunk.text) for chunk in chunks],
    )
    return len(chunks)


def embed_missing(
    connection: sqlite3.Connection,
    model: str,
    announce: Callable[[str, int, int], None],
    batch_size: int = 64,
) -> None:
    """Embed every passage that has no vector yet, batched across notes.

    One request per note meant 452 round trips and nearly four minutes on this
    vault. Batching turns it into 41 requests. Leaving the vectors until after
    the rows are written also makes an interrupted first run resumable: the
    passages are already there and only the gaps get filled.
    """
    rows = connection.execute(
        "SELECT chunks.id, chunks.heading, chunks.text, notes.title "
        "FROM chunks JOIN notes ON notes.path = chunks.path WHERE chunks.vector IS NULL"
    ).fetchall()

    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        texts = [embedding_text(row["title"], row["heading"], row["text"]) for row in batch]
        vectors = ollama.embed_documents(texts, model)
        connection.executemany(
            "UPDATE chunks SET vector = ? WHERE id = ?",
            [(pack(vector), row["id"]) for row, vector in zip(batch, vectors)],
        )
        connection.commit()
        announce("embedding passages", min(start + batch_size, len(rows)), len(rows))


def rewrite_links(connection: sqlite3.Connection, notes: list[Note]) -> None:
    """Rebuild every edge, because a new note can resolve links that were dead.

    Obsidian links by note name, so `[[Alpha]]` points nowhere until some note
    called Alpha exists, and then it points there from everywhere at once.
    """
    by_name: dict[str, str] = {}
    for note in notes:
        by_name.setdefault(Path(note.path).stem.lower(), note.path)
        by_name.setdefault(note.path.lower(), note.path)
        by_name.setdefault(note.path.lower().removesuffix(".md"), note.path)

    edges = {
        (note.path, by_name[target.lower().removesuffix(".md")])
        for note in notes
        for target in note.links
        if target.lower().removesuffix(".md") in by_name
        and by_name[target.lower().removesuffix(".md")] != note.path
    }
    connection.execute("DELETE FROM links")
    connection.executemany("INSERT INTO links(source, target) VALUES (?, ?)", sorted(edges))


def pack(vector: list[float]) -> bytes:
    """Store unit vectors, so a dot product is already the cosine."""
    array = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(array))
    return (array if length == 0 else array / length).tobytes()


def counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        name: connection.execute(f"SELECT COUNT(*) AS total FROM {name}").fetchone()["total"]
        for name in ("notes", "chunks", "links")
    }
