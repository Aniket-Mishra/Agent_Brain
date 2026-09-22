"""Agent Brain: chat with an Obsidian vault using local models.

Run it with: streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from brain import index, ollama
from brain.locate import database_path, find_vault, is_vault, remember_vault
from views import browse, chat, graph

DEFAULT_CONTEXT = 8192
EMBEDDING_HINTS = ("embed", "bge", "minilm")

st.set_page_config(page_title="Agent Brain", page_icon="◐", layout="wide")


@st.cache_resource(show_spinner=False)
def open_index(database: str) -> sqlite3.Connection:
    """One connection for the session. Streamlit re-runs this script constantly."""
    return index.connect(Path(database))


@st.cache_data(ttl=30, show_spinner=False)
def list_models() -> list[str]:
    return ollama.installed_models()


def main() -> None:
    vault = resolve_vault()
    if vault is None:
        return

    models = list_models()
    if not models:
        st.error("Cannot reach Ollama. Start it with `ollama serve`, then reload.")
        return

    connection = open_index(str(database_path(vault)))

    # Reserved so the vault summary sits at the top, even though its numbers
    # are only correct after any re-index below has finished.
    summary = st.sidebar.container()

    embedding_model = pick_model(models, EMBEDDING_HINTS, "Embedding model", "embedding_model")
    chat_model = pick_model(models, (), "Chat model", "chat_model", avoid=EMBEDDING_HINTS)
    passages, context_size = show_tuning(connection, vault)

    keep_index_current(connection, vault, embedding_model)
    with summary:
        show_summary(connection, vault)

    chat_tab, browse_tab, graph_tab = st.tabs(["Chat", "Browse", "Graph"])
    with chat_tab:
        chat.render(connection, chat_model, embedding_model, passages, context_size)
    with browse_tab:
        browse.render(connection, vault)
    with graph_tab:
        graph.render(connection)


def resolve_vault() -> Path | None:
    """Find the vault, and ask once if it cannot be found.

    The answer is remembered outside the repository, so a clone on another
    machine with a different folder layout only asks once.
    """
    found = find_vault()
    if found is not None:
        return found

    st.title("Agent Brain")
    st.write("Point me at your Obsidian vault. I look for the folder containing `.obsidian`.")
    typed = st.text_input("Vault path", placeholder="~/Notes")
    if not typed:
        return None

    candidate = Path(typed).expanduser().resolve()
    if not is_vault(candidate):
        st.error(f"No `.obsidian` folder in {candidate}")
        return None

    remember_vault(candidate)
    st.rerun()
    return None


def pick_model(
    models: list[str], hints: tuple[str, ...], label: str, key: str, avoid: tuple[str, ...] = ()
) -> str:
    """Offer only models Ollama already has, defaulting to a sensible one."""
    preferred = [name for name in models if matches(name, hints, avoid)] or models
    remembered = st.session_state.get(key)
    start = preferred.index(remembered) if remembered in preferred else 0
    chosen = st.sidebar.selectbox(label, preferred, index=start, key=key)
    return chosen


def matches(name: str, hints: tuple[str, ...], avoid: tuple[str, ...]) -> bool:
    lowered = name.lower()
    if any(word in lowered for word in avoid):
        return False
    return not hints or any(word in lowered for word in hints)


def keep_index_current(connection: sqlite3.Connection, vault: Path, embedding_model: str) -> None:
    """Re-index when the vault has moved on, and only then.

    The check compares paths and modification times, so it costs a directory
    walk rather than reading 452 files on every interaction.
    """
    if index.is_current(connection, vault, embedding_model):
        return

    status = st.status("Updating the index", expanded=True)
    progress = status.progress(0.0)

    def announce(stage: str, done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0), text=f"{stage}  {done}/{total}")

    try:
        report = index.refresh(connection, vault, embedding_model, announce)
    except ollama.OllamaError as error:
        status.update(label=str(error), state="error")
        st.stop()

    status.update(
        label=f"Indexed {report.added} new, {report.updated} changed, "
        f"{report.removed} removed in {report.seconds:.0f}s",
        state="complete",
        expanded=False,
    )


def show_summary(connection: sqlite3.Connection, vault: Path) -> None:
    totals = index.counts(connection)
    st.markdown(f"### {vault.name}")
    st.caption(f"{totals['notes']} notes · {totals['chunks']} passages · {totals['links']} links")
    st.divider()


def show_tuning(connection: sqlite3.Connection, vault: Path) -> tuple[int, int]:
    """Settings worth changing rarely, kept out of the way."""
    with st.sidebar.expander("Settings"):
        passages = st.slider("Passages in context", 3, 15, 8)
        context_size = st.select_slider(
            "Context window", [4096, 8192, 16384, 32768], value=DEFAULT_CONTEXT
        )
        st.caption(f"Vault: `{vault}`")
        st.caption(f"Index: `{database_path(vault)}`")
        if st.button("Rebuild index from scratch", use_container_width=True):
            connection.execute("DELETE FROM notes")
            connection.execute("DELETE FROM meta WHERE key = 'vault_fingerprint'")
            connection.commit()
            st.rerun()
    return passages, context_size


main()
