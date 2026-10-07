import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def cli(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    for name, kind, text in [
        ("Focus", "research", "SQLite stores local passages. [[Linked]]"),
        ("Linked", "research", "The index supports recovery after interruption."),
        ("Hidden", "daily", "Pineapple secret belongs to this daily note."),
    ]:
        (vault / f"{name}.md").write_text(f"---\ntype: {kind}\ncreated: 2026-03-12\n---\n# Decision\n{text}\n")
    before = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in vault.glob("*.md")}
    sent = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            sent.append((self.path, body))
            if self.server.fail and self.path == "/api/chat":
                payload = {"error": "synthetic model failure"}
            elif self.path == "/api/tags":
                payload = {"models": [{"name": "nomic-embed-text:latest", "digest": "embed-v1"}]}
            elif self.path == "/api/embed":
                payload = {"embeddings": [[1.0, 0.0] for _text in body["input"]]}
            elif self.path == "/api/chat":
                count = sum(message["role"] == "tool" for message in body["messages"])
                if count < 2:
                    name, arguments = ("search", {"query": "SQLite"}) if not count else ("read", {"path": "Focus.md"})
                    message = {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}
                else:
                    message = {"content": "SQLite stores local passages [E1]."}
                payload = {"message": message, "prompt_eval_count": 40, "eval_count": 8}
            else:
                payload = {"capabilities": ["completion", "tools"]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.fail = False
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        environment = {
            **os.environ, "AGENT_BRAIN_VAULT": str(vault), "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "OLLAMA_HOST": f"http://127.0.0.1:{server.server_port}",
            "AGENT_BRAIN_CHAT_MODEL": "local-chat", "AGENT_BRAIN_EMBED_MODEL": "nomic-embed-text",
        }

        def run(*arguments, **overrides):
            return subprocess.run(
                [sys.executable, "ask.py", *arguments], cwd=Path(__file__).resolve().parents[1],
                env={**environment, **overrides}, capture_output=True, text=True, timeout=20,
            )

        try:
            yield run, sent, server, vault
        finally:
            server.shutdown()
            worker.join()
            assert {path.name: hashlib.sha256(path.read_bytes()).digest() for path in vault.glob("*.md")} == before


def test_cli_indexes_searches_reads_surveys_and_answers(cli) -> None:
    run, sent, _server, _vault = cli
    result = run("search", "SQLite", "--kind", "research", "--since", "2026-03-01", "--until", "2026-03-31")
    assert result.returncode == 0, result.stderr
    assert "Focus.md" in result.stdout and "SQLite" in result.stdout
    assert "Hidden.md" not in result.stdout and "Pineapple" not in result.stdout
    assert "reading notes" in result.stderr and "embedding passages" in result.stderr
    assert "embedding passages" not in result.stdout
    count = len(sent)
    result = run("read", "Focus.md", OLLAMA_HOST="http://127.0.0.1:1")
    assert result.returncode == 0 and "SQLite" in result.stdout and not result.stderr
    result = run("survey", "--kind", "research", OLLAMA_HOST="http://127.0.0.1:1")
    assert result.returncode == 0 and "Notes: 2" in result.stdout and not result.stderr
    assert len(sent) == count
    result = run("ask", "What", "stores", "passages?", "--max-tokens", "123", "--timeout", "5")
    assert result.returncode == 0, result.stderr
    assert "SQLite stores local passages [E1]." in result.stdout and "Focus.md" in result.stdout
    assert "tool search" in result.stderr and "tool read" in result.stderr
    assert "tool search" not in result.stdout
    chats = [body for route, body in sent if route == "/api/chat"]
    assert len(chats) == 3 and chats[0]["messages"][-1]["content"] == "What stores passages?"
    assert all(body["options"]["num_predict"] == 123 for body in chats)
    assert any("[E1]" in message.get("content", "") for message in chats[-1]["messages"] if message["role"] == "tool")


@pytest.mark.parametrize("arguments,error", [
    ((), "command"), (("search",), "question"), (("ask", ""), "Question must not be empty"),
    (("ask", "SQLite", "--prior-user", "Earlier"), "Supply --prior-user and --prior-assistant together"),
    (("survey", "--since", "2026-02-30"), "date must be YYYY-MM-DD"),
    (("survey", "--since", "2026-04-01", "--until", "2026-03-01"), "--since must not be after --until"),
])
def test_cli_bad_arguments_fail_before_models(cli, arguments, error) -> None:
    run, sent, _server, _vault = cli
    result = run(*arguments)
    assert result.returncode == 2 and error in result.stderr and not result.stdout
    assert not sent


def test_invalid_explicit_vault_does_not_fall_back_to_remembered_vault(cli) -> None:
    run, sent, _server, vault = cli
    cache = vault.parent / "cache" / "agent_brain"
    cache.mkdir(parents=True)
    (cache / "vault.json").write_text(json.dumps({"vault": str(vault)}))
    result = run("survey", AGENT_BRAIN_VAULT=str(vault.parent / "missing"))
    assert result.returncode == 2 and ".obsidian" in result.stderr
    assert not sent and not list(cache.glob("*.db"))


def test_cli_model_failure_exits_nonzero_without_answer(cli) -> None:
    run, _sent, server, _vault = cli
    assert run("search", "SQLite").returncode == 0
    server.fail = True
    result = run("ask", "What stores passages?")
    assert result.returncode == 1 and "synthetic model failure" in result.stderr
    assert not result.stdout
