"""Talking to a local Ollama server, using only the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Several embedding models are trained with asymmetric prefixes and quietly lose
# recall without them. Keeping the mapping here rather than in config means it
# cannot be set wrong.
PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


class OllamaError(RuntimeError):
    pass


def installed_models() -> list[str]:
    """Model names Ollama already has, so the UI can offer real choices."""
    try:
        payload = request("/api/tags", None)
    except OllamaError:
        return []
    return sorted(model["name"] for model in payload.get("models", []))


def embed_documents(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []
    _query_prefix, document_prefix = PREFIXES.get(base_name(model), ("", ""))
    payload = request(
        "/api/embed",
        {"model": model, "input": [document_prefix + text for text in texts]},
    )
    return payload.get("embeddings", [])


def embed_query(text: str, model: str) -> list[float]:
    query_prefix, _document_prefix = PREFIXES.get(base_name(model), ("", ""))
    payload = request("/api/embed", {"model": model, "input": [query_prefix + text]})
    embeddings = payload.get("embeddings", [])
    if not embeddings:
        raise OllamaError(f"{model} returned no embedding")
    return embeddings[0]


def stream_reply(messages: list[dict[str, str]], model: str, context_size: int) -> Iterator[str]:
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": context_size},
    }
    request_object = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request_object, timeout=600) as response:
            yield from read_events(response)
    except urllib.error.URLError as error:
        raise OllamaError(explain(error, model)) from error


def read_events(response) -> Iterator[str]:
    """Ollama sends one JSON object per line while it generates."""
    for line in response:
        if not line.strip():
            continue
        event = json.loads(line)
        piece = event.get("message", {}).get("content", "")
        if piece:
            yield piece
        if event.get("done"):
            return


def request(route: str, body: dict | None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_object = urllib.request.Request(
        f"{HOST}{route}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request_object, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.URLError as error:
        raise OllamaError(explain(error, body.get("model", "") if body else "")) from error


def base_name(model: str) -> str:
    return model.split(":", 1)[0]


def canonical(model: str) -> str:
    """`nomic-embed-text` and `nomic-embed-text:latest` name the same model.

    Ollama's tag list returns the tagged form while a command line or config
    usually carries the bare one. Treating them as different models re-embeds
    the whole vault for nothing.
    """
    return model if ":" in model else f"{model}:latest"


def explain(error: urllib.error.URLError, model: str) -> str:
    status = getattr(error, "code", None)
    if status == 404:
        return f"Ollama does not have {model!r}. Run: ollama pull {model}"
    if isinstance(error, urllib.error.HTTPError):
        return f"Ollama returned {status}: {error.read().decode('utf-8', 'replace')[:200]}"
    return f"Cannot reach Ollama at {HOST}. Start it with: ollama serve"
