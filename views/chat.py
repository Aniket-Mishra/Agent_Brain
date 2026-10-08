"""Fixed search chat, with an opt-in loop for exploring indexed notes."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import streamlit as st

from brain import loop, ollama
from brain.answer import NOTHING_FOUND, build_messages, source_budget
from brain.search import Hit, Scope, search


def render(
    connection: sqlite3.Connection,
    chat_model: str,
    embedding_model: str,
    passages: int,
    context_size: int,
    num_predict: int,
    scope: Scope,
    vault_identity: str,
) -> None:
    history = st.session_state.setdefault("history", [])
    context = (vault_identity, scope)
    if st.session_state.get("chat_context") != context:
        st.session_state.pop("chat_prior", None)
        st.session_state["chat_context"] = context
    experimental = st.checkbox(
        "Explore notes with tools (experimental)", key="explore_with_tools",
        help=(
            "Allows extra searches and link navigation. "
            "Fixed search chat remains the default while answer quality is evaluated."
        ),
    )

    for item in history:
        with st.chat_message(item["role"]):
            if item.get("turn") is not None:
                show_turn(item["turn"])
            else:
                st.markdown(item["text"])
                if item.get("sources"):
                    show_sources(item["sources"])

    if not scope.is_everything:
        st.caption("Answering from the filtered notes only. Clear the filters to widen it.")

    question = st.chat_input("Ask your vault something", submit_mode="disable")
    if not question:
        return

    history.append({"role": "user", "text": question, "context": context})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = None
        try:
            if experimental:
                status = st.status("Exploring notes", expanded=False)
                result = loop.answer(
                    connection, question, prior=st.session_state.get("chat_prior"),
                    chat_model=chat_model, embed_model=embedding_model,
                    passages=passages, context_size=context_size, scope=scope,
                    limits=loop.Limits(num_predict=num_predict),
                    prior_scope=scope,
                    progress=lambda name: status.update(label=f"Exploring notes: {name}"),
                )
                status.update(
                    label="Exploration complete" if result.status != "limit" else "Exploration stopped at a limit",
                    state="complete" if result.status != "limit" else "error",
                )
                show_turn(result)
                history.append({
                    "role": "assistant", "text": result.text, "turn": result,
                    "context": context,
                })
                if result.status == "answered":
                    st.session_state["chat_prior"] = (question, result.text)
                return
            with st.spinner("Searching your notes"):
                hits = search(
                    connection, question, embedding_model, limit=passages, scope=scope
                )
            if not hits:
                st.markdown(NOTHING_FOUND)
                history.append({"role": "assistant", "text": NOTHING_FOUND, "context": context})
                return
            messages = build_messages(question, hits, source_budget(context_size, num_predict))
            answer = st.write_stream(ollama.stream_reply(
                messages, chat_model, context_size, num_predict=num_predict,
            ))
        except (ollama.OllamaError, OSError, sqlite3.Error, ValueError) as error:
            if status is not None:
                status.update(label="Exploration failed", state="error")
            st.error(str(error))
            return
        show_sources(hits)

    history.append({
        "role": "assistant", "text": answer, "sources": hits,
        "context": context,
    })
    st.session_state["chat_prior"] = (question, answer)


def show_turn(turn: loop.Turn) -> None:
    st.markdown(turn.text)
    if turn.status == "limit":
        st.caption("Incomplete result")
    show_sources(turn.evidence, turn.cited)
    if turn.facts:
        with st.expander(f"{len(turn.facts)} computed index results"):
            for number, fact in enumerate(turn.facts, 1):
                state = "cited" if number in turn.cited_facts else "retrieved"
                st.markdown(f"**[I{number}] {state}**")
                st.text(fact.text)


def show_sources(hits: Sequence[Hit], cited: tuple[int, ...] | None = None) -> None:
    if not hits:
        return
    label = f"{len(hits)} sources" if cited is None else f"{len(hits)} retrieved passages"
    with st.expander(label):
        for number, hit in enumerate(hits, start=1):
            identifier = f"[{number}]" if cited is None else f"[E{number}]"
            state = "" if cited is None else (" · cited" if number in cited else " · retrieved")
            st.markdown(f"**{identifier} {hit.title}{state}**")
            details = [hit.path]
            if hit.heading:
                details.append(hit.heading)
            details.extend(part for part in (hit.kind, hit.created) if part)
            if hit.found_by:
                details.append(f"found by {', '.join(sorted(hit.found_by))}")
            st.caption("  ·  ".join(details))
            st.text(hit.text)
