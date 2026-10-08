from datetime import date
from pathlib import Path
from shutil import copy2

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from brain import index, locate, loop, ollama, themes, tools
from brain.search import Hit, Scope

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_backend(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    for name, kind, created, body in [
        ("Focus", "research", "2026-03-10", "SQLite stores local passages on this machine."),
        ("Peer", "research", "2026-03-11", "Rebuild the index after switching embedding models."),
        ("Daily", "daily", "2026-03-12", "This daily note records a separate task."),
    ]:
        (vault / f"{name}.md").write_text(f"---\ntype: {kind}\ncreated: {created}\n---\n# Decision\n{body}\n")
    state = {
        "vault": vault, "status": "answered", "error": False,
        "loop_calls": [], "stream_calls": [], "theme_calls": [],
        "database_path": locate.database_path,
    }
    monkeypatch.delenv("AGENT_BRAIN_VAULT", raising=False)
    monkeypatch.delenv("AGENT_BRAIN_CHAT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_BRAIN_EMBED_MODEL", raising=False)
    monkeypatch.setattr(locate, "find_vault", lambda: state["vault"])
    monkeypatch.setattr(locate, "database_path", lambda _vault: tmp_path / "index.db")
    monkeypatch.setattr(ollama, "local_host", lambda: "http://localhost:11434")
    monkeypatch.setattr(ollama, "installed_models", lambda: ["llama:latest", "nomic-embed-text:latest", "qwen3:4b"])
    monkeypatch.setattr(ollama, "model_digest", lambda _model: "local-embed-digest")
    monkeypatch.setattr(ollama, "embed_documents", lambda texts, _model: [[1.0, 0.0] for _text in texts])
    monkeypatch.setattr(ollama, "embed_query", lambda *_args: [1.0, 0.0])

    def no_network(*_args, **_kwargs):
        pytest.fail("AppTest attempted a live Ollama request")

    monkeypatch.setattr(ollama, "request", no_network)

    def stream(messages, model, context_size, *, num_predict=800):
        state["stream_calls"].append((messages, num_predict))
        yield "SQLite stores local passages [1]."

    def answer(connection, question, **options):
        state["loop_calls"].append((question, options))
        if state["error"]:
            raise ollama.OllamaError("synthetic model failure")
        options["progress"]("read")
        hits = (
            Hit(1, "Focus.md", "Storage choice", "Storage > Decision", "x" * 1600 + " visible tail [cut]", "research", "2026-03-10", found_by={"read"}),
            Hit(2, "Peer.md", "Recovery plan", "Recovery", "Rebuild the index.", "research", "2026-03-11", found_by={"words"}),
        )
        return loop.Turn(
            f"Answer to {question} [E1].", status=state["status"], evidence=hits, cited=(1,),
            facts=(tools.ToolResult("Notes: 2", paths=("Focus.md", "Peer.md")),), cited_facts=(1,),
            tool_names=("read",), model_calls=2,
        )

    def find_themes(connection, model, count=8, scope=None):
        state["theme_calls"].append((model, count, scope))
        return [themes.Theme("Storage", ["Focus.md", "Peer.md"])]

    monkeypatch.setattr(ollama, "stream_reply", stream)
    monkeypatch.setattr(loop, "answer", answer)
    monkeypatch.setattr(themes, "find_themes", find_themes)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield state
    st.cache_data.clear()
    st.cache_resource.clear()


def test_search_stays_default_and_tool_mode_renders_admitted_sources(app_backend, monkeypatch) -> None:
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    assert app.selectbox(key="chat").value == "qwen3:4b"
    assert app.selectbox(key="embedding").value == "nomic-embed-text:latest"
    assert not app.checkbox(key="explore_with_tools").value
    assert app.number_input(key="generation_tokens").value == loop.Limits().num_predict
    app.number_input(key="generation_tokens").set_value(1536).run()
    app.chat_input[0].set_value("SQLite storage question").run()
    assert not app.exception and len(app_backend["stream_calls"]) == 1
    assert app_backend["stream_calls"][-1][1] == 1536
    assert not app_backend["loop_calls"]
    assert any("[1] Focus" in element.value for element in app.markdown)
    app.checkbox(key="explore_with_tools").check().run()
    app.chat_input[0].set_value("Follow up").run()
    assert not app.exception
    assert app_backend["loop_calls"][-1][1]["prior"] == ("SQLite storage question", "SQLite stores local passages [1].")
    assert app_backend["loop_calls"][-1][1]["limits"].num_predict == 1536
    shown = "\n".join(element.value for element in app.markdown)
    assert "[E1] Storage choice · cited" in shown and "[E2] Recovery plan · retrieved" in shown
    assert "[I1] cited" in shown
    assert any("visible tail [cut]" in element.value for element in app.text)
    assert any("Storage > Decision" in element.value and "found by read" in element.value for element in app.caption)
    assert app.session_state["history"][-1]["turn"].evidence[0].title == "Storage choice"
    calls = (len(app_backend["stream_calls"]), len(app_backend["loop_calls"]))
    for bad_value in (128.5, 4097):
        monkeypatch.setattr(st, "number_input", lambda *_args, **_kwargs: bad_value)
        app.run()
        assert not app.exception and not app.chat_input
        assert app.error and "Generation tokens must be an integer from 128 to 4096" in app.error[0].value
        assert (len(app_backend["stream_calls"]), len(app_backend["loop_calls"])) == calls


def test_prior_survives_good_filtered_turns_and_resets_on_scope_or_vault_change(app_backend) -> None:
    app = AppTest.from_file(APP).run()
    app.checkbox(key="explore_with_tools").check().run()
    app.multiselect(key="scope_kinds").set_value(["research"]).run()
    app.chat_input[0].set_value("First question").run()
    app.chat_input[0].set_value("Second question").run()
    assert not app.exception
    options = app_backend["loop_calls"][-1][1]
    assert options["prior"] == ("First question", "Answer to First question [E1].")
    assert options["prior_scope"] == Scope(kinds=("research",))
    app_backend["status"] = "limit"
    app.chat_input[0].set_value("Limited question").run()
    app_backend["error"] = True
    app.chat_input[0].set_value("Broken question").run()
    assert not app.exception and app.error
    app_backend["status"], app_backend["error"] = "answered", False
    app.chat_input[0].set_value("Next question").run()
    assert app_backend["loop_calls"][-1][1]["prior"] == ("Second question", "Answer to Second question [E1].")
    app.multiselect(key="scope_kinds").set_value(["daily"]).run()
    app.chat_input[0].set_value("Daily question").run()
    assert app_backend["loop_calls"][-1][1]["prior"] is None
    other = app_backend["vault"].parent / "other"
    (other / ".obsidian").mkdir(parents=True)
    (other / "Other.md").write_text("# Other\nA different vault contains a different note.\n")
    app_backend["vault"] = other
    app.run()
    app.chat_input[0].set_value("Other question").run()
    assert not app.exception and app_backend["loop_calls"][-1][1]["prior"] is None
    assert any("Answer to First question" in element.value for element in app.markdown)


def test_themes_are_invalidated_by_controls_and_index_changes(app_backend) -> None:
    app = AppTest.from_file(APP).run()

    def find():
        next(button for button in app.button if button.label == "Find themes").click().run()
        assert not app.exception and app.session_state["themes"]

    find()
    app.slider(key="theme_groups").set_value(5).run()
    assert not app.exception and "themes" not in app.session_state
    find()
    app.multiselect(key="scope_kinds").set_value(["research"]).run()
    assert "themes" not in app.session_state
    find()
    app.selectbox(key="chat").select("llama:latest").run()
    assert "themes" not in app.session_state
    find()
    other = app_backend["vault"].parent / "identical-vault"
    (other / ".obsidian").mkdir(parents=True)
    for path in app_backend["vault"].glob("*.md"):
        copy2(path, other / path.name)
    app_backend["vault"] = other
    app.run()
    assert not app.exception and "themes" not in app.session_state
    find()
    (app_backend["vault"] / "Focus.md").write_text("# Changed\nSQLite remains the local passage store after this edit.\n")
    app.run()
    assert not app.exception and "themes" not in app.session_state
    assert len(app_backend["theme_calls"]) == 5


def test_dates_work_without_types_and_changed_vault_stops_chat(app_backend, monkeypatch) -> None:
    for path in app_backend["vault"].glob("*.md"):
        path.write_text(path.read_text().replace("type: research\n", "").replace("type: daily\n", ""))
    app = AppTest.from_file(APP).run()
    assert not app.exception and not app.multiselect(key="scope_kinds").options
    app.checkbox(key="limit_date").check().run()
    app.date_input(key="scope_dates").set_value((date(2026, 3, 11), date(2026, 3, 11))).run()
    assert not app.exception
    assert app.session_state["chat_context"][1] == Scope(since="2026-03-11", until="2026-03-11")
    refreshes = []
    monkeypatch.setattr(index, "is_current", lambda *_args, **_kwargs: False)

    def refresh(*_args, **options):
        refreshes.append(options)
        return index.Report()

    monkeypatch.setattr(index, "refresh", refresh)
    app.run()
    assert not app.exception and not app.chat_input
    assert refreshes == [{"model_digest": "local-embed-digest"}]
    assert app.warning and "changed while indexing" in app.warning[0].value


def test_cache_inside_vault_stops_app_with_configuration_error(app_backend, monkeypatch) -> None:
    monkeypatch.setattr(locate, "CACHE_DIRECTORY", app_backend["vault"] / ".cache")
    monkeypatch.setattr(locate, "database_path", app_backend["database_path"])
    app = AppTest.from_file(APP).run()
    assert not app.exception and not app.chat_input
    assert app.error and "Set XDG_CACHE_HOME outside the vault" in app.error[0].value
    assert not (app_backend["vault"] / ".cache").exists()
