"""Agent Brain: chat with an Obsidian vault using local models.

Run it with: streamlit run app.py
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import streamlit as st

from brain import index, links, loop, ollama
from brain.locate import database_path, find_vault, is_vault, remember_vault
from brain.search import Scope
from views import browse, chat, explore, graph

DEFAULT_CONTEXT = 8192
EMBEDDING_HINTS = ("embed", "bge", "minilm")

st.set_page_config(page_title="Agent Brain", page_icon="◐", layout="wide")


@st.cache_resource(
    show_spinner=False, scope="session", max_entries=1,
    on_release=lambda connection: connection.close(),
)
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

    try:
        ollama.local_host()
        connection = open_index(str(database_path(vault)))
    except (ollama.OllamaError, OSError, sqlite3.Error, ValueError) as error:
        st.error(str(error))
        st.stop()
    models = list_models()
    if not models:
        st.error("Local models unavailable. Start Ollama and pull nomic-embed-text and qwen3:4b, then reload.")
        return

    # Reserved so the vault summary sits at the top, even though its numbers
    # are only correct after any re-index below has finished.
    summary = st.sidebar.container()

    embedding_models, chat_models = split_models(models)
    preferred_embed = index.read_meta(connection, "embedding_model") or os.environ.get(
        "AGENT_BRAIN_EMBED_MODEL", "nomic-embed-text"
    )
    preferred_chat = os.environ.get("AGENT_BRAIN_CHAT_MODEL", "qwen3:4b")
    vault_identity = str(vault.resolve())
    if st.session_state.get("model_vault") != vault_identity:
        st.session_state.pop("embedding", None)
        st.session_state["model_vault"] = vault_identity
    embedding_model = st.sidebar.selectbox(
        "Embedding model", embedding_models,
        index=model_position(embedding_models, preferred_embed), key="embedding",
    )
    chat_model = st.sidebar.selectbox(
        "Chat model", chat_models, index=model_position(chat_models, preferred_chat), key="chat",
    )
    passages, context_size, num_predict = show_settings(vault)

    keep_index_current(connection, vault, embedding_model)
    with summary:
        show_summary(connection, vault)
    scope = show_scope(connection)

    chat_tab, browse_tab, graph_tab, explore_tab = st.tabs(
        ["Chat", "Browse", "Graph", "Explore"]
    )
    with chat_tab:
        chat.render(
            connection, chat_model, embedding_model, passages, context_size,
            num_predict, scope, vault_identity,
        )
    with browse_tab:
        st.caption("Browse shows the full vault. Search filters apply to Chat and themes.")
        browse.render(connection, vault)
    with graph_tab:
        st.caption("Graph shows the full vault. Search filters apply to Chat and themes.")
        graph.render(connection)
    with explore_tab:
        explore.render(connection, chat_model, scope, vault_identity)


def resolve_vault() -> Path | None:
    """Find the vault, and ask once if it cannot be found.

    The answer is remembered outside the repository, so a clone on another
    machine with a different folder layout only asks once.
    """
    found = find_vault()
    if found is not None:
        return found
    if os.environ.get("AGENT_BRAIN_VAULT"):
        st.error("AGENT_BRAIN_VAULT must point to a folder containing `.obsidian`. Correct it, then reload.")
        return None

    st.title("Agent Brain")
    st.write("Point me at your Obsidian vault. I look for the folder containing `.obsidian`.")
    typed = st.text_input("Vault path", placeholder="~/Notes")
    if not typed:
        return None

    candidate = Path(typed).expanduser().resolve()
    if not is_vault(candidate):
        st.error(f"No `.obsidian` folder in {candidate}")
        return None

    try:
        remember_vault(candidate)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return None
    st.rerun()
    return None


def split_models(models: list[str]) -> tuple[list[str], list[str]]:
    """Populate model pickers using names.

    Name hints avoid an /api/show request for every installed model. Offer all
    models when those hints have no matches.
    """
    embedding = [name for name in models if any(hint in name.lower() for hint in EMBEDDING_HINTS)]
    chatting = [name for name in models if name not in set(embedding)]
    return embedding or models, chatting or models


def model_position(models: list[str], preferred: str) -> int:
    return next(
        (
            position for position, model in enumerate(models)
            if ollama.canonical(model) == ollama.canonical(preferred)
        ),
        0,
    )


def keep_index_current(connection: sqlite3.Connection, vault: Path, embedding_model: str) -> None:
    """Refresh changed notes and refuse partial or stale index snapshots."""
    status = None
    try:
        digest = ollama.model_digest(embedding_model)
        if index.is_current(connection, vault, embedding_model, model_digest=digest):
            return
        status = st.status("Updating the index", expanded=True)
        progress = status.progress(0.0)

        def announce(stage: str, done: int, total: int) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"{stage}  {done}/{total}")

        report = index.refresh(connection, vault, embedding_model, announce, model_digest=digest)
        if not index.is_current(connection, vault, embedding_model, model_digest=digest):
            status.update(label="Vault changed while indexing", state="error")
            st.warning("The vault changed while indexing. Reload to finish updating before asking questions.")
            st.stop()
    except (ollama.OllamaError, OSError, sqlite3.Error, ValueError) as error:
        if status is not None:
            status.update(label="Index update failed", state="error")
        st.error(str(error))
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
    if st.button("Rebuild index", width="stretch"):
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
    st.sidebar.markdown("**Search in**")
    chosen = st.sidebar.multiselect(
        "Note types",
        [kind for kind, _total in available],
        format_func=lambda kind: f"{kind} ({dict(available)[kind]})",
        placeholder="every type",
        key="scope_kinds",
    )

    first, last = links.date_range(connection)
    since = until = ""
    if first and st.sidebar.checkbox("Limit by date", key="limit_date"):
        start, end = date.fromisoformat(first), date.fromisoformat(last)
        picked = st.sidebar.date_input(
            "Between", value=(start, end), min_value=start, max_value=end, key="scope_dates"
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


def show_settings(vault: Path) -> tuple[int, int, int]:
    """Settings worth changing rarely, kept out of the way."""
    with st.sidebar.expander("Settings"):
        passages = st.slider("Passages in context", 3, 15, 8)
        context_size = st.select_slider(
            "Context window", [4096, 8192, 16384, 32768], value=DEFAULT_CONTEXT
        )
        num_predict = st.number_input(
            "Generation tokens", 128, 4096, loop.Limits().num_predict, step=128,
            key="generation_tokens", help="Includes model thinking. Higher limits can take longer.",
        )
        if type(num_predict) is not int or not 128 <= num_predict <= 4096:
            st.error("Generation tokens must be an integer from 128 to 4096.")
            st.stop()
        st.caption(f"Vault: `{vault}`")
        st.caption(f"Index: `{database_path(vault)}`")
    return passages, context_size, num_predict


main()
