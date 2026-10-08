"""Talking to a local Ollama server, using only the standard library."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str = "stop"
    thinking: str = ""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def local_host() -> str:
    host = HOST if "://" in HOST else f"http://{HOST}"
    try:
        parsed = urllib.parse.urlsplit(host)
        name = parsed.hostname or ""
        parsed.port
        loopback = name == "localhost" or ipaddress.ip_address(name).is_loopback
    except ValueError as error:
        raise OllamaError("Set OLLAMA_HOST to a valid loopback Ollama URL") from error
    if (
        not loopback or parsed.scheme not in {"http", "https"}
        or parsed.username or parsed.password or parsed.path not in {"", "/"}
        or parsed.query or parsed.fragment
    ):
        raise OllamaError("Set OLLAMA_HOST to a loopback Ollama URL, such as http://localhost:11434")
    return host.rstrip("/")


def model_details(model: str) -> dict:
    details = request("/api/show", {"model": model}, timeout=10)
    if details.get("remote_model") or details.get("remote_host"):
        raise OllamaError(f"Cloud model {model!r} is disabled. Choose a local model.")
    return details


def model_digest(model: str) -> str:
    for item in request("/api/tags", None, timeout=10).get("models", []):
        if canonical(item["name"]) == canonical(model):
            if item.get("remote_model") or item.get("remote_host"):
                raise OllamaError(f"Cloud model {model!r} is disabled. Choose a local model.")
            return item["digest"]
    raise OllamaError(f"Ollama does not have {model!r}. Run: ollama pull {model}")


def installed_models() -> list[str]:
    """Model names Ollama already has, so the UI can offer real choices."""
    try:
        payload = request("/api/tags", None)
    except OllamaError:
        return []
    return sorted(
        model["name"] for model in payload.get("models", [])
        if not model.get("remote_model") and not model.get("remote_host")
    )


def embed_documents(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []
    model_details(model)
    _query_prefix, document_prefix = PREFIXES.get(base_name(model), ("", ""))
    payload = request(
        "/api/embed",
        {"model": model, "input": [document_prefix + text for text in texts], "truncate": False},
    )
    return payload.get("embeddings", [])


def embed_query(text: str, model: str) -> list[float]:
    model_details(model)
    query_prefix, _document_prefix = PREFIXES.get(base_name(model), ("", ""))
    payload = request("/api/embed", {"model": model, "input": [query_prefix + text], "truncate": False})
    embeddings = payload.get("embeddings", [])
    if not embeddings:
        raise OllamaError(f"{model} returned no embedding")
    vector = embeddings[0]
    if not isinstance(vector, list) or not vector or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise OllamaError(f"{model} returned an invalid query embedding")
    return vector


def complete(
    messages: list[dict], model: str, context_size: int, tools: list[dict] | None = None,
    *, num_predict: int = 800, timeout: float = 120,
) -> Reply:
    details = model_details(model)
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Native parsing separates reasoning from the answer on thinking models.
        "think": "thinking" in details.get("capabilities", []),
        "truncate": False,
        "shift": False,
        "options": {"temperature": 0.2, "num_ctx": context_size, "num_predict": num_predict},
    }
    if tools is not None:
        body["tools"] = tools
    payload = request("/api/chat", body, timeout=timeout)
    if payload.get("done") is not True:
        raise OllamaError("Ollama response ended before a complete answer")
    return parse_reply(payload)


def parse_reply(payload: dict) -> Reply:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama returned no message")
    content = message.get("content", "")
    thinking = message.get("thinking", "")
    calls = message.get("tool_calls", [])
    if not isinstance(content, str) or not isinstance(thinking, str) or not isinstance(calls, list):
        raise OllamaError("Ollama returned an invalid message")
    parsed = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise OllamaError("Ollama returned an invalid tool call")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise OllamaError("Ollama returned invalid tool arguments") from error
        if not isinstance(arguments, dict):
            raise OllamaError("Ollama returned invalid tool arguments")
        parsed.append(ToolCall(function["name"], arguments))
    try:
        json.dumps(
            [content, thinking, [(call.name, call.arguments) for call in parsed]], ensure_ascii=False,
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise OllamaError("Ollama returned invalid UTF-8 text") from error
    return Reply(
        content, tuple(parsed), payload.get("prompt_eval_count", 0),
        payload.get("eval_count", 0), payload.get("done_reason", "stop"), thinking,
    )


def opener():
    # A loopback destination must not send note text through an HTTP proxy or redirect.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def request(route: str, body: dict | None, *, timeout: float = 600) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_object = urllib.request.Request(
        f"{local_host()}{route}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with opener().open(request_object, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise OllamaError(explain(error, body.get("model", "") if body else "")) from error
    except json.JSONDecodeError as error:
        raise OllamaError("Ollama returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise OllamaError("Ollama returned an invalid response")
    if payload.get("error"):
        raise OllamaError(str(payload["error"]))
    return payload


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
