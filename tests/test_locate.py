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
