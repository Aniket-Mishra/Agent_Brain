import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from brain import index, ollama, themes


@pytest.mark.parametrize("payload,expected", [
    ({"message": {"content": '"Storage"'}, "done": True}, "Storage"),
    ({"message": {"content": ""}, "done": True}, "Daily"),
    ({"message": {"content": "Partial"}, "done": True, "done_reason": "length"}, "Daily"),
    ({"message": {"content": "Partial"}, "done": False}, "Daily"),
    ({"message": {"content": "Partial"}}, "Daily"),
    ({"error": "model failed"}, "Daily"),
])
def test_indexed_theme_naming_uses_completed_http_reply(tmp_path, monkeypatch, payload, expected):
    vault = tmp_path / "vault"
    folder = vault / "Daily"
    folder.mkdir(parents=True)
    for name in ("Alpha", "Beta", "Gamma"):
        (folder / f"{name}.md").write_text(f"# Storage\n{name} records a local SQLite storage decision.\n")
    before = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in folder.glob("*.md")}
    sent = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sent.append((self.path, body))
            if self.path == "/api/embed":
                response = {"embeddings": [[1.0, 0.0] for _text in body["input"]]}
            elif self.path == "/api/chat":
                response = payload
            else:
                response = {"capabilities": ["completion"]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        monkeypatch.setattr(ollama, "HOST", f"http://127.0.0.1:{server.server_port}")
        connection = index.connect(tmp_path / "index.db")
        try:
            index.refresh(connection, vault, "nomic-embed-text")
            assert index.is_current(connection, vault, "nomic-embed-text")
            found = themes.find_themes(connection, "local", count=1)
            assert len(found) == 1 and found[0].name == expected
            assert found[0].paths == ["Daily/Alpha.md", "Daily/Beta.md", "Daily/Gamma.md"]
            chats = [body for route, body in sent if route == "/api/chat"]
            assert len(chats) == 1 and chats[0]["stream"] is False
            assert chats[0]["options"]["num_ctx"] == 4096
            assert chats[0]["options"]["num_predict"] == 800
            assert "tools" not in chats[0]
            assert {path.name: hashlib.sha256(path.read_bytes()).digest() for path in folder.glob("*.md")} == before
        finally:
            connection.close()
            server.shutdown()
            worker.join()
