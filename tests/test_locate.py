from brain import locate
import pytest


def test_explicit_invalid_vault_does_not_fall_back(tmp_path, monkeypatch) -> None:
    other_vault = tmp_path / "other"
    (other_vault / ".obsidian").mkdir(parents=True)
    monkeypatch.setenv("AGENT_BRAIN_VAULT", str(tmp_path / "missing"))
    monkeypatch.setattr(locate, "read_remembered_vault", lambda: other_vault)
    monkeypatch.setattr(locate, "sibling_vaults", lambda: [other_vault])
    assert locate.find_vault() is None


def test_cache_configuration_cannot_write_inside_vault(tmp_path, monkeypatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    cache = vault / ".cache"
    monkeypatch.setattr(locate, "CACHE_DIRECTORY", cache)
    monkeypatch.setattr(locate, "REMEMBERED", cache / "vault.json")
    for operation in (locate.database_path, locate.remember_vault):
        with pytest.raises(ValueError, match="outside the vault"):
            operation(vault)
    assert list(vault.iterdir()) == []


@pytest.mark.parametrize("operation", [locate.database_path, locate.remember_vault])
def test_cache_destination_symlink_cannot_write_into_vault(tmp_path, monkeypatch, operation) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("Original note content.", encoding="utf-8")
    original = note.read_bytes()
    cache = tmp_path / "cache"
    monkeypatch.setattr(locate, "CACHE_DIRECTORY", cache)
    monkeypatch.setattr(locate, "REMEMBERED", cache / "vault.json")
    database = locate.database_path(vault)
    destination = database if operation is locate.database_path else locate.REMEMBERED
    destination.symlink_to(note)

    with pytest.raises(ValueError, match="outside the vault"):
        operation(vault)
    assert note.read_bytes() == original
    assert destination.is_symlink()


def test_external_cache_still_remembers_vault_and_returns_database(tmp_path, monkeypatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setattr(locate, "CACHE_DIRECTORY", cache)
    monkeypatch.setattr(locate, "REMEMBERED", cache / "vault.json")

    database = locate.database_path(vault)
    locate.remember_vault(vault)
    assert database.parent == cache and database.suffix == ".db"
    assert locate.read_remembered_vault() == vault
    assert list(vault.iterdir()) == []
