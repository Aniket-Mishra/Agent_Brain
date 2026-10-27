import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from brain import ollama


def test_native_reply_preserves_calls_and_usage() -> None:
    reply = ollama.parse_reply({
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "SQLite"}}},
            {"function": {"name": "read", "arguments": '{"path":"Storage.md"}'}},
        ]}, "prompt_eval_count": 15, "eval_count": 4, "done_reason": "stop",
    })
    assert reply.tool_calls == (
        ollama.ToolCall("search", {"query": "SQLite"}),
        ollama.ToolCall("read", {"path": "Storage.md"}),
    )
    assert (reply.prompt_eval_count, reply.eval_count, reply.done_reason) == (15, 4, "stop")


def test_native_thinking_is_separate_from_the_displayed_answer(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "model_details", lambda _model: {"capabilities": ["thinking"]})
    sent = []

    def request(_route, body, **_options):
        sent.append(body)
        return {"message": {"thinking": "Private reasoning [E99]", "content": "Answer [E1]"}, "done": True}

    monkeypatch.setattr(ollama, "request", request)
    reply = ollama.complete([{"role": "user", "content": "Question"}], "local", 4096)
    assert sent[0]["think"] is True
    assert reply.content == "Answer [E1]"
    assert reply.thinking == "Private reasoning [E99]"


@pytest.mark.parametrize("arguments", [{"query": "\ud800"}, '{"query":"\\ud800"}'])
def test_invalid_unicode_in_model_arguments_is_a_transport_failure(arguments) -> None:
    with pytest.raises(ollama.OllamaError, match="UTF-8"):
        ollama.parse_reply({"message": {"tool_calls": [
            {"function": {"name": "search", "arguments": arguments}},
        ]}})


@pytest.mark.parametrize("message", [None, [], {"content": 7}, {"tool_calls": {}}, {"tool_calls": [None]}])
def test_malformed_native_message_has_transport_error(message) -> None:
    with pytest.raises(ollama.OllamaError):
        ollama.parse_reply({"message": message})


def test_malformed_survey_arguments_are_not_converted_to_valid_call() -> None:
    for arguments in ("broken-json", "[]", "null", "false", [], None, False):
        with pytest.raises(ollama.OllamaError, match="invalid tool arguments"):
            ollama.parse_reply({"message": {"tool_calls": [
                {"function": {"name": "survey", "arguments": arguments}},
            ]}})


def test_survey_with_missing_arguments_is_valid() -> None:
    reply = ollama.parse_reply({"message": {"tool_calls": [{"function": {"name": "survey"}}]}})
    assert reply.tool_calls == (ollama.ToolCall("survey", {}),)


def test_completion_omits_tools_and_forwards_execution_limits(monkeypatch) -> None:
    sent = []

    def request(route, body, **options):
        sent.append((route, body, options))
        return {} if route == "/api/show" else {"message": {"content": "Answer"}, "done": True}

    monkeypatch.setattr(ollama, "request", request)
    assert ollama.complete([], "local", 4096, num_predict=123, timeout=17).content == "Answer"
    route, body, options = sent[-1]
    assert route == "/api/chat" and "tools" not in body
    assert body["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 123}
    assert body["truncate"] is False and body["shift"] is False
    assert body["stream"] is False
    assert options == {"timeout": 17}


@pytest.mark.parametrize("done", [None, False, 0, 1, "true"])
def test_completion_rejects_missing_or_invalid_done(monkeypatch, done) -> None:
    payload = {"message": {"content": "Partial answer"}}
    if done is not None:
        payload["done"] = done
    monkeypatch.setattr(ollama, "request", lambda route, *_args, **_options: {} if route == "/api/show" else payload)
    with pytest.raises(ollama.OllamaError, match="complete answer"):
        ollama.complete([], "local", 4096)


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], [True], ["1"], None])
def test_query_rejects_nonfinite_or_malformed_embeddings(monkeypatch, vector) -> None:
    monkeypatch.setattr(ollama, "request", lambda route, *_args, **_kwargs: {} if route == "/api/show" else {"embeddings": [vector]})
    with pytest.raises(ollama.OllamaError, match="invalid query embedding"):
        ollama.embed_query("SQLite", "local")


def test_embeddings_disable_truncation_and_use_model_prefix(monkeypatch) -> None:
    sent = []

    def request(route, body, **_options):
        sent.append((route, body))
        return {} if route == "/api/show" else {"embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr(ollama, "request", request)
    ollama.embed_documents(["decision"], "nomic-embed-text")
    ollama.embed_query("SQLite", "nomic-embed-text")
    bodies = [body for route, body in sent if route == "/api/embed"]
    assert [body["input"] for body in bodies] == [["search_document: decision"], ["search_query: SQLite"]]
    assert all(body["truncate"] is False for body in bodies)


@pytest.mark.parametrize("field", ["remote_model", "remote_host"])
def test_cloud_advertisement_rejected_before_content(monkeypatch, field) -> None:
    sent = []

    def request(route, body, **_options):
        sent.append((route, body))
        return {field: "cloud", "models": [{"name": "local:latest", "digest": "v1", field: "cloud"}]}

    monkeypatch.setattr(ollama, "request", request)
    with pytest.raises(ollama.OllamaError, match="Cloud model"):
        ollama.complete([{"role": "user", "content": "private note"}], "local", 4096)
    with pytest.raises(ollama.OllamaError, match="Cloud model"):
        ollama.embed_documents(["private note"], "local")
    with pytest.raises(ollama.OllamaError, match="Cloud model"):
        ollama.model_digest("local")
    assert ollama.installed_models() == []
    assert all(route in {"/api/show", "/api/tags"} for route, _body in sent)
    assert "private note" not in json.dumps(sent)


@pytest.mark.parametrize("host", ["http://10.0.0.1:11434", "http://example.com", "http://user@localhost:11434", "http://localhost/api", "http://localhost:invalid", "http://localhost:65536", "http://localhost?proxy=1"])
def test_endpoint_must_be_plain_valid_loopback(monkeypatch, host) -> None:
    monkeypatch.setattr(ollama, "HOST", host)
    with pytest.raises(ollama.OllamaError, match="loopback"):
        ollama.local_host()


def test_transport_bypasses_proxy_and_refuses_redirect(monkeypatch) -> None:
    reached = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/received")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b"invalid" if self.path == "/invalid-json" else b"[]" if self.path == "/invalid-shape"
                    else b'{"error":"model failed"}' if self.path == "/model-error" else b'{"ok":true}'
                )

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        monkeypatch.setattr(ollama, "HOST", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("no_proxy", "")
        try:
            assert ollama.request("/plain", None, timeout=2) == {"ok": True}
            for route in ("/invalid-json", "/invalid-shape"):
                with pytest.raises(ollama.OllamaError, match="invalid"):
                    ollama.request(route, None, timeout=2)
            with pytest.raises(ollama.OllamaError, match="model failed"):
                ollama.request("/model-error", None, timeout=2)
            with pytest.raises(ollama.OllamaError, match="302"):
                ollama.request("/redirect", None, timeout=2)
            assert reached == ["/plain", "/invalid-json", "/invalid-shape", "/model-error", "/redirect"]
        finally:
            server.shutdown()
            worker.join()
