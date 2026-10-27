"""Where things live on this machine: which folder is the vault, and where the index goes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIRECTORY = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "agent_brain"
REMEMBERED = CACHE_DIRECTORY / "vault.json"


def is_vault(candidate: Path) -> bool:
    return candidate.is_dir() and (candidate / ".obsidian").is_dir()


def find_vault() -> Path | None:
    """Locate the vault without being told, so a fresh clone just runs.

    Order: an explicit environment variable, the folder chosen last time, a
    single sibling of this repository, then any parent of the current folder.
    Returns None when it cannot tell, and the caller asks.
    """
    from_environment = os.environ.get("AGENT_BRAIN_VAULT")
    if from_environment:
        named = Path(from_environment).expanduser()
        if is_vault(named):
            return named.resolve()
        return None

    remembered = read_remembered_vault()
    if remembered and is_vault(remembered):
        return remembered

    siblings = sibling_vaults()
    # Two vaults side by side is a coin flip, and picking the wrong one silently
    # is worse than one question.
    if len(siblings) == 1:
        return siblings[0]

    for parent in [Path.cwd(), *Path.cwd().parents]:
        if is_vault(parent):
            return parent
    return None


def sibling_vaults() -> list[Path]:
    repository = Path(__file__).resolve().parent.parent
    try:
        neighbours = sorted(repository.parent.iterdir())
    except OSError:
        return []
    return [path for path in neighbours if path != repository and is_vault(path)]


def remember_vault(vault: Path) -> None:
    prepare_cache(vault, REMEMBERED)
    REMEMBERED.write_text(json.dumps({"vault": str(vault)}), encoding="utf-8")


def read_remembered_vault() -> Path | None:
    if not REMEMBERED.is_file():
        return None
    try:
        return Path(json.loads(REMEMBERED.read_text(encoding="utf-8"))["vault"])
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def database_path(vault: Path) -> Path:
    """Keep one derived index per vault in the external cache."""
    fingerprint = hashlib.blake2b(str(vault).encode(), digest_size=6).hexdigest()
    database = CACHE_DIRECTORY / f"{vault.name}-{fingerprint}.db"
    prepare_cache(vault, database)
    return database


def prepare_cache(vault: Path, destination: Path) -> None:
    root = vault.resolve()
    if CACHE_DIRECTORY.resolve().is_relative_to(root) or destination.resolve().is_relative_to(root):
        raise ValueError("Set XDG_CACHE_HOME outside the vault; its files are read only")
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
