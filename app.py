"""Agent Brain: chat with an Obsidian vault using local models.

Run it with: streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import streamlit as st

from brain import index, links, ollama
from brain.locate import database_path, find_vault, is_vault, remember_vault
from brain.search import Scope
from views import browse, chat, explore, graph

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

    embedding_models, chat_models = split_models(models)
    embedding_model = st.sidebar.selectbox("Embedding model", embedding_models, key="embedding")
    chat_model = st.sidebar.selectbox("Chat model", chat_models, key="chat")
    passages, context_size = show_settings(vault)

    keep_index_current(connection, vault, embedding_model)
    with summary:
        show_summary(connection, vault)
    scope = show_scope(connection)

    chat_tab, browse_tab, graph_tab, explore_tab = st.tabs(
        ["Chat", "Browse", "Graph", "Explore"]
    )
    with chat_tab:
        chat.render(connection, chat_model, embedding_model, passages, context_size, scope)
    with browse_tab:
        browse.render(connection, vault)
    with graph_tab:
        graph.render(connection)
    with explore_tab:
        explore.render(connection, chat_model, scope)


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


def split_models(models: list[str]) -> tuple[list[str], list[str]]:
    """Separate embedding models from chat models.

    Ollama does not say which is which, so the name is the only signal. If the
    guess comes up empty, offer everything rather than an empty dropdown.
    """
    embedding = [name for name in models if any(hint in name.lower() for hint in EMBEDDING_HINTS)]
    chatting = [name for name in models if name not in set(embedding)]
    return embedding or models, chatting or models


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
    if st.button("Rebuild index", use_container_width=True):
        index.clear(connection)
        st.rerun()
    st.divider()


def show_scope(connection: sqlite3.Connection) -> Scope:
    """Narrow what gets searched, using the frontmatter the notes already carry.

    A date filter beats parsing dates out of the question. "March 2026" as free
    text matched the words March and 2026 and returned January notes; as a
    range it cannot be wrong.
    """
    available = links.kinds(connection)
    if not available:
        return Scope()

    st.sidebar.markdown("**Search in**")
    chosen = st.sidebar.multiselect(
        "Note types",
        [kind for kind, _total in available],
        format_func=lambda kind: f"{kind} ({dict(available)[kind]})",
        placeholder="every type",
    )

    first, last = links.date_range(connection)
    since = until = ""
    if first and st.sidebar.checkbox("Limit by date"):
        start, end = date.fromisoformat(first), date.fromisoformat(last)
        picked = st.sidebar.date_input(
            "Between", value=(start, end), min_value=start, max_value=end
        )
        # While the user is mid-selection Streamlit hands back a single date.
        if isinstance(picked, tuple) and len(picked) == 2:
            since, until = picked[0].isoformat(), picked[1].isoformat()

    scope = Scope(kinds=tuple(chosen), since=since, until=until)
    if not scope.is_everything:
        st.sidebar.caption(f"{scoped_note_count(connection, scope)} notes in scope")
    st.sidebar.divider()
    return scope


def scoped_note_count(connection: sqlite3.Connection, scope: Scope) -> int:
    where, values = scope.clause()
    return connection.execute(
        f"SELECT COUNT(*) AS total FROM notes WHERE {where}", values
    ).fetchone()["total"]


def show_settings(vault: Path) -> tuple[int, int]:
    """Settings worth changing rarely, kept out of the way."""
    with st.sidebar.expander("Settings"):
        passages = st.slider("Passages in context", 3, 15, 8)
        context_size = st.select_slider(
            "Context window", [4096, 8192, 16384, 32768], value=DEFAULT_CONTEXT
        )
        st.caption(f"Vault: `{vault}`")
        st.caption(f"Index: `{database_path(vault)}`")
    return passages, context_size


main()
