"""A SQLite passage cache with vectors and the vault's link graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain import ollama
from brain.notes import (
    Note,
    chunk_note,
    date_from_name,
    embedding_text,
    find_notes,
    read_note,
)

# INDEX_VERSION rebuilds unchanged files when parsing or vector provenance changes.
SCHEMA_VERSION = "3"
INDEX_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    path    TEXT PRIMARY KEY,
    title   TEXT NOT NULL,
    digest  TEXT NOT NULL,
    size    INTEGER NOT NULL DEFAULT 0,
    kind    TEXT NOT NULL DEFAULT '',
    tags    TEXT NOT NULL DEFAULT '[]',
    created TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS notes_by_kind ON notes(kind);
CREATE INDEX IF NOT EXISTS notes_by_created ON notes(created);

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
    target TEXT,
    raw    TEXT NOT NULL
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
    # `with sqlite3.connect(...)` opens a transaction, not a closing scope, so
    # the connection has to be closed by hand or it leaks on every call.
    try:
        probe = sqlite3.connect(database)
    except sqlite3.DatabaseError:
        return None
    try:
        row = probe.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return row[0] if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        probe.close()


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
        (path.relative_to(vault).as_posix(), path.stat().st_mtime_ns, path.stat().st_size)
        for path in find_notes(vault)
    )
    return hashlib.blake2b(repr(entries).encode("utf-8"), digest_size=8).hexdigest()


def is_current(
    connection: sqlite3.Connection,
    vault: Path,
    model: str,
    *,
    model_digest: str | None = None,
) -> bool:
    if read_meta(connection, "index_version") != INDEX_VERSION:
        return False
    if not same_model(read_meta(connection, "embedding_model"), model):
        return False
    if model_digest is not None and read_meta(connection, "embedding_digest") != model_digest:
        return False
    if connection.execute("SELECT 1 FROM chunks WHERE vector IS NULL LIMIT 1").fetchone():
        return False
    return read_meta(connection, "vault_fingerprint") == vault_fingerprint(vault)


def same_model(stored: str | None, wanted: str) -> bool:
    return stored is not None and ollama.canonical(stored) == ollama.canonical(wanted)


def refresh(
    connection: sqlite3.Connection,
    vault: Path,
    model: str,
    progress: Callable[[str, int, int], None] | None = None,
    *,
    model_digest: str | None = None,
) -> Report:
    """Bring the index in line with what is on disk, re-embedding only what changed."""
    started = time.perf_counter()
    report = Report()
    announce = progress or (lambda *_: None)

    stored_model = read_meta(connection, "embedding_model")
    outdated = read_meta(connection, "index_version") != INDEX_VERSION
    model_changed = stored_model is not None and not same_model(stored_model, model)
    digest_changed = (
        model_digest is not None and read_meta(connection, "embedding_digest") != model_digest
    )
    with connection:
        connection.execute("DELETE FROM meta WHERE key = 'vault_fingerprint'")
        if outdated or model_changed or digest_changed:
            report.rebuilt = bool(connection.execute("SELECT 1 FROM notes LIMIT 1").fetchone())
            connection.execute("DELETE FROM notes")
            connection.execute("DELETE FROM links")
        # Batch commits must carry their model even when a later batch fails.
        write_meta(connection, "embedding_model", model)
        write_meta(connection, "index_version", INDEX_VERSION)
        if model_digest is None:
            connection.execute("DELETE FROM meta WHERE key = 'embedding_digest'")
        else:
            write_meta(connection, "embedding_digest", model_digest)

    starting_fingerprint = vault_fingerprint(vault)
    known = {
        row["path"]: row["digest"]
        for row in connection.execute("SELECT path, digest FROM notes")
    }
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

    # Links are written before the slow step. They cost milliseconds and do not
    # depend on any vector, so doing them last meant an interrupted embed left
    # the table emptied and not refilled, and the graph blank until the next run.
    rewrite_links(connection, notes)
    connection.commit()

    embed_missing(connection, model, announce)
    if starting_fingerprint == vault_fingerprint(vault):
        write_meta(connection, "vault_fingerprint", starting_fingerprint)
    write_meta(connection, "indexed_at", time.time())
    connection.commit()

    report.seconds = time.perf_counter() - started
    return report


def store_note(connection: sqlite3.Connection, note: Note) -> int:
    """Write the note and its passages, leaving the vectors for the batch pass."""
    connection.execute("DELETE FROM notes WHERE path = ?", (note.path,))
    connection.execute(
        "INSERT INTO notes(path, title, digest, size, kind, tags, created) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            note.path,
            note.title,
            note.digest,
            note.size,
            note.kind,
            json.dumps(note.tags),
            note.created or date_from_name(note.path),
        ),
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
    widths = {
        row[0] for row in connection.execute(
            "SELECT DISTINCT length(vector) FROM chunks WHERE vector IS NOT NULL"
        )
    }
    if len(widths) > 1 or any(width == 0 or width % 4 for width in widths):
        raise ollama.OllamaError("The index contains incompatible embedding dimensions")
    width = next(iter(widths), 0) // 4

    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        texts = [embedding_text(row["title"], row["heading"], row["text"]) for row in batch]
        vectors = validate_embeddings(ollama.embed_documents(texts, model), len(batch), width)
        width = vectors.shape[1]
        connection.executemany(
            "UPDATE chunks SET vector = ? WHERE id = ?",
            [(pack(vector), row["id"]) for row, vector in zip(batch, vectors)],
        )
        connection.commit()
        announce("embedding passages", min(start + batch_size, len(rows)), len(rows))


def validate_embeddings(vectors: list[list[float]], count: int, width: int) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ollama.OllamaError("Ollama returned malformed embeddings") from error
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0:
        raise ollama.OllamaError("Ollama returned the wrong number or shape of embeddings")
    if width and matrix.shape[1] != width:
        raise ollama.OllamaError("Ollama changed embedding dimensions during indexing")
    if not np.isfinite(matrix).all():
        raise ollama.OllamaError("Ollama returned non-finite embeddings")
    return matrix


def rewrite_links(connection: sqlite3.Connection, notes: list[Note]) -> None:
    """Rebuild every edge, because a new note can resolve links that were dead.

    Obsidian links by note name, so `[[Alpha]]` points nowhere until some note
    called Alpha exists, and then it points there from everywhere at once.
    """
    by_name: dict[str, str] = {}
    for note in notes:
        by_name.setdefault(link_key(Path(note.path).stem), note.path)
        by_name.setdefault(link_key(note.path), note.path)

    edges: set[tuple[str, str | None, str]] = set()
    for note in notes:
        edges |= edges_from(note, by_name)

    connection.execute("DELETE FROM links")
    connection.executemany(
        "INSERT INTO links(source, target, raw) VALUES (?, ?, ?)",
        sorted(edges, key=lambda edge: (edge[0], edge[2])),
    )


def link_key(name: str) -> str:
    """How a link target and a note name are compared: lower case, no extension."""
    return name.lower().removesuffix(".md")


def edges_from(note: Note, by_name: dict[str, str]) -> set[tuple[str, str | None, str]]:
    """Every link out of a note, including the ones pointing at nothing.

    An unresolved link is kept with a null target because it is a signal, not
    an error: a concept linked eight times and never written is worth knowing.
    """
    found = set()
    for target in note.links:
        destination = by_name.get(link_key(target))
        if destination == note.path:
            continue
        found.add((note.path, destination, target))
    return found


def pack(vector: list[float] | np.ndarray) -> bytes:
    """Store unit vectors, so a dot product is already the cosine."""
    array = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(array))
    normalized = array if length == 0 else array / length
    return normalized.astype(np.float32).tobytes()


def all_paths(connection: sqlite3.Connection) -> list[str]:
    """Every indexed note, for the pickers in the UI."""
    return [row["path"] for row in connection.execute("SELECT path FROM notes ORDER BY path")]


def counts(connection: sqlite3.Connection) -> dict[str, int]:
    # The table names are literals from the tuple below, never user input.
    return {
        name: connection.execute(f"SELECT COUNT(*) AS total FROM {name}").fetchone()["total"]
        for name in ("notes", "chunks", "links")
    }


def clear(connection: sqlite3.Connection) -> None:
    """Drop everything derived from the vault so the next launch rebuilds it."""
    connection.execute("DELETE FROM notes")
    connection.execute("DELETE FROM meta WHERE key = 'vault_fingerprint'")
    connection.commit()
