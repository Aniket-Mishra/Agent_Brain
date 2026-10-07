"""Recovery and read boundaries, using temporary vaults and stub embeddings."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from brain import index, ollama
from brain.notes import date_from_name, find_notes, read_date, read_note
from brain.search import Scope, search, similar_notes


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    for number in range(70):
        (root / f"{number:02d}.md").write_text(
            f"This note records storage decision number {number}.\n", encoding="utf-8"
        )
    return root


@pytest.fixture
def connection(tmp_path: Path):
    database = index.connect(tmp_path / "index.db")
    yield database
    database.close()


def stored_vectors(connection) -> np.ndarray:
    rows = connection.execute("SELECT vector FROM chunks WHERE vector IS NOT NULL").fetchall()
    return np.vstack([np.frombuffer(row[0], dtype=np.float32) for row in rows])


def test_interrupted_model_switch_cannot_mix_vectors(connection, vault, monkeypatch) -> None:
    calls = {"model-a": 0, "model-b": 0}

    def embed(texts, model):
        calls[model] += 1
        if model == "model-b" and calls[model] == 2:
            raise ollama.OllamaError("interrupted batch")
        vector = [1.0, 0.0] if model == "model-a" else [0.0, 1.0]
        return [vector] * len(texts)

    monkeypatch.setattr(ollama, "embed_documents", embed)
    index.refresh(connection, vault, "model-a")
    with pytest.raises(ollama.OllamaError, match="interrupted batch"):
        index.refresh(connection, vault, "model-b")

    assert index.read_meta(connection, "embedding_model") == "model-b"
    assert index.read_meta(connection, "vault_fingerprint") is None
    assert not index.is_current(connection, vault, "model-a")
    assert not index.is_current(connection, vault, "model-b")
    assert len(stored_vectors(connection)) == 64

    index.refresh(connection, vault, "model-a")
    assert calls["model-a"] == 4
    assert np.array_equal(stored_vectors(connection), np.tile([1.0, 0.0], (70, 1)))
    assert index.is_current(connection, vault, "model-a")


def test_failed_batch_resumes_only_missing_vectors(connection, vault, monkeypatch) -> None:
    calls = []

    def embed(texts, model):
        calls.append(len(texts))
        if len(calls) == 2:
            raise ollama.OllamaError("interrupted batch")
        return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(ollama, "embed_documents", embed)
    with pytest.raises(ollama.OllamaError):
        index.refresh(connection, vault, "model-a")
    index.refresh(connection, vault, "model-a")

    assert calls == [64, 6, 6]
    assert index.is_current(connection, vault, "model-a")


def test_replaced_model_digest_rebuilds_same_name(connection, vault, monkeypatch) -> None:
    calls = []

    def embed(texts, model):
        calls.append(len(texts))
        return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(ollama, "embed_documents", embed)
    index.refresh(connection, vault, "model-a", model_digest="first-digest")
    assert index.is_current(connection, vault, "model-a", model_digest="first-digest")
    assert not index.is_current(connection, vault, "model-a", model_digest="new-digest")
    assert index.refresh(connection, vault, "model-a", model_digest="new-digest").rebuilt
    assert index.read_meta(connection, "embedding_digest") == "new-digest"
    assert calls == [64, 6, 64, 6]
    assert index.is_current(connection, vault, "model-a", model_digest="new-digest")


def test_edit_during_embedding_leaves_index_stale(connection, vault, monkeypatch) -> None:
    path = vault / "00.md"
    edited = False

    def embed(texts, model):
        nonlocal edited
        if not edited:
            previous = path.stat()
            path.write_text("Changed decision: use Postgres instead of storage.\n", "utf-8")
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
            edited = True
        return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(ollama, "embed_documents", embed)
    index.refresh(connection, vault, "model-a")
    assert not index.is_current(connection, vault, "model-a")
    assert index.read_meta(connection, "vault_fingerprint") is None

    assert index.refresh(connection, vault, "model-a").updated == 1
    assert index.is_current(connection, vault, "model-a")
    assert "Postgres" in connection.execute(
        "SELECT text FROM chunks WHERE path = '00.md'"
    ).fetchone()[0]


@pytest.mark.parametrize(
    "vectors",
    [[], [[1.0]], [[1.0]] * 65, [[]] * 64, [[1.0], [1.0, 2.0]] * 32,
     [[float("nan")]] * 64, [[float("inf")]] * 64, None],
)
def test_malformed_batch_writes_no_vectors(connection, vault, monkeypatch, vectors) -> None:
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: vectors)
    with pytest.raises(ollama.OllamaError):
        index.refresh(connection, vault, "model-a")
    assert connection.execute("SELECT 1 FROM chunks WHERE vector IS NOT NULL").fetchone() is None
    assert not index.is_current(connection, vault, "model-a")


def test_resumed_embedding_dimensions_must_match(connection, vault, monkeypatch) -> None:
    def first_attempt(texts, model):
        if len(texts) == 6:
            raise ollama.OllamaError("interrupted batch")
        return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(ollama, "embed_documents", first_attempt)
    with pytest.raises(ollama.OllamaError):
        index.refresh(connection, vault, "model-a")
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))

    with pytest.raises(ollama.OllamaError, match="dimensions"):
        index.refresh(connection, vault, "model-a")
    assert len(stored_vectors(connection)) == 64
    assert not index.is_current(connection, vault, "model-a")


def test_large_finite_vectors_normalize_without_overflow(connection, vault, monkeypatch) -> None:
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[3e38, 3e38]] * len(texts))
    index.refresh(connection, vault, "model-a")
    assert np.allclose(np.linalg.norm(stored_vectors(connection), axis=1), 1.0)


def test_cache_contract_invalidates_old_dates_and_vectors(connection, vault, monkeypatch) -> None:
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))
    index.refresh(connection, vault, "model-a")
    connection.execute("DELETE FROM meta WHERE key = 'index_version'")
    connection.execute("UPDATE notes SET created = '2026-02-31'")
    connection.commit()

    assert not index.is_current(connection, vault, "model-a")
    assert index.refresh(connection, vault, "model-a").rebuilt
    assert connection.execute("SELECT 1 FROM notes WHERE created != ''").fetchone() is None
    assert index.read_meta(connection, "schema_version") == "3"


def test_missing_vector_never_reports_current(connection, vault, monkeypatch) -> None:
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))
    index.refresh(connection, vault, "model-a")
    connection.execute("UPDATE chunks SET vector = NULL WHERE path = '00.md'")
    connection.commit()
    assert not index.is_current(connection, vault, "model-a")


def test_vault_reads_skip_external_symlinks_and_accept_internal(tmp_path, vault) -> None:
    external = tmp_path / "private.md"
    external.write_text("Private text outside the vault.\n", "utf-8")
    (vault / "external.md").symlink_to(external)
    (vault / "internal.md").symlink_to(vault / "00.md")

    assert vault / "external.md" not in find_notes(vault)
    assert vault / "internal.md" in find_notes(vault)
    with pytest.raises(ValueError, match="outside the vault"):
        read_note(vault / "external.md", vault)
    with pytest.raises(ValueError, match="outside the vault"):
        read_note(vault / ".." / "private.md", vault)
    assert read_note(vault / "internal.md", vault).path == "internal.md"


@pytest.mark.parametrize("raw", ["2026-02-31", "31st February 2026", "2025-02-29", "0 Jan 2026"])
def test_invalid_calendar_dates_are_undated(raw: str) -> None:
    assert read_date({"date_created": raw}) == ""
    assert date_from_name(f"daily/{raw}.md") == ""
    assert read_date({"date_created": raw, "created": "2024-02-29"}) == "2024-02-29"


def test_until_only_filter_excludes_undated_notes(connection, vault, monkeypatch) -> None:
    (vault / "00.md").write_text(
        "---\ndate: 2026-01-01\n---\nStorage decision inside date scope.\n", "utf-8"
    )
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))
    monkeypatch.setattr(ollama, "embed_query", lambda text, model: [1.0])
    index.refresh(connection, vault, "model-a")

    hits = search(connection, "storage decision", "model-a", scope=Scope(until="2026-02-01"))
    assert [hit.path for hit in hits] == ["00.md"]


def test_related_scope_applies_before_limit(connection, vault, monkeypatch) -> None:
    (vault / "69.md").write_text(
        "---\ntype: research\n---\nStorage decision in the last research note.\n", "utf-8"
    )
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, model: [[1.0]] * len(texts))
    index.refresh(connection, vault, "model-a")

    assert similar_notes(connection, "00.md", limit=1) == [("01.md", 1.0)]
    assert similar_notes(connection, "00.md", limit=1, scope=Scope(kinds=("research",))) == [
        ("69.md", 1.0)
    ]
