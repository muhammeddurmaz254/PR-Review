"""Stage [4]: the only module that talks to a model server.

Ollama's ``/api/chat`` takes a JSON schema in ``format`` and constrains decoding
to it, which is the same guarantee vLLM gives through guided decoding -- so the
rest of the pipeline never learns which server it is running against, and the
serving decision stays open until phase 0b measures both.

Two stubs implement the same interface. They are not test scaffolding: `silent`
and `flag_all` are the honest lower and upper bounds of any detector built on
this candidate set, and running them costs nothing.

Only the standard library is used. ``requirements.txt`` is deliberately two
packages long, and a review bot that needs an HTTP client for one POST does not
justify a third.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import contract

DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class Response:
    """One model reply plus what it cost.

    ``prompt_tokens`` comes from the server, so the context budget in the run
    manifest is measured rather than estimated.
    """

    text: str
    # What the model produced before the answer, when thinking is on. Ollama
    # returns it in its own field, so a schema still constrains the answer --
    # the reason this was disabled from the first model run onward was wrong.
    thinking: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    error: str = ""
    raw: dict = field(default_factory=dict)


class Client(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict) -> Response: ...


@dataclass
class OllamaClient:
    """Chat completion against an Ollama server, local or tunnelled.

    ``keep_alive`` holds the weights resident between calls; on a rented card the
    default five minutes would unload the model during a slow corpus run and pay
    the load cost again.
    """

    model: str
    base_url: str = DEFAULT_BASE_URL
    num_ctx: int = 8192
    temperature: float = 0.0
    seed: int = 7
    timeout: float = 300.0
    keep_alive: str = "30m"
    retries: int = 2
    # Reasoning models emit a thinking block before the answer, which a schema
    # cannot constrain and a JSON parser cannot read. Left unset for every other
    # model, because Ollama rejects the key when the model has no such mode.
    think: bool | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def _headers(self) -> dict[str, str]:
        # A free ngrok tunnel answers browser-looking requests with an HTML
        # interstitial instead of the API, which would surface as a JSON decode
        # error against a server that is in fact healthy.
        return {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}

    def _payload(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def complete(self, system: str, user: str, schema: dict) -> Response:
        body = json.dumps(self._payload(system, user, schema)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=body, headers=self._headers(), method="POST",
        )
        last = ""
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    document = json.loads(handle.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                last = f"{type(error).__name__}: {error}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                continue
            message = document.get("message", {})
            return Response(
                text=message.get("content", ""),
                thinking=message.get("thinking", "") or "",
                prompt_tokens=int(document.get("prompt_eval_count", 0)),
                completion_tokens=int(document.get("eval_count", 0)),
                duration_ms=int((time.monotonic() - started) * 1000),
                raw=document,
            )
        return Response(text="", error=last)

    def chat(self, messages: list[dict], tools: list | None = None, schema: dict | None = None) -> dict:
        """One turn of a conversation, with tools offered or a schema imposed.

        The agent loop in `detect/agent.py` needs the raw message back -- a
        tool call is not text -- so this returns the document Ollama sent,
        or ``{"error": ...}`` after the same retries ``complete`` makes.
        """
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        if schema is not None:
            payload["format"] = schema
        if self.think is not None:
            payload["think"] = self.think
        request = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/chat",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers=self._headers(), method="POST")
        last = ""
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                # The body says why: a 500 from a failed tool-call parse and one
                # from an exhausted card look identical without it.
                body = error.read().decode("utf-8", "replace")[:500]
                last = f"HTTPError {error.code}: {body}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                last = f"{type(error).__name__}: {error}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        return {"error": last}

    def build(self) -> str:
        """The served model's digest and quantization, or "" if unknown.

        A resumed run can be stitched from two machines -- this corpus has been
        served from a Kaggle notebook and a Colab one on the same tunnel URL --
        and answers from two different builds are not one measurement. The digest
        is what makes that checkable instead of assumed.
        """
        probe = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags", headers=self._headers())
        try:
            with urllib.request.urlopen(probe, timeout=30) as handle:
                document = json.loads(handle.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""
        for entry in document.get("models", []):
            if entry.get("name") == self.model or entry.get("name", "").startswith(f"{self.model}:"):
                details = entry.get("details", {})
                return f"{entry.get('digest', '')[:16]}/{details.get('quantization_level', '?')}"
        return ""

    def health(self) -> str:
        """Which models the server has, so a typo fails before the corpus runs."""
        probe = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags", headers=self._headers())
        try:
            with urllib.request.urlopen(probe, timeout=30) as handle:
                document = json.loads(handle.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            return f"unreachable: {type(error).__name__}: {error}"
        names = [entry.get("name", "") for entry in document.get("models", [])]
        if not any(name == self.model or name.startswith(f"{self.model}:") for name in names):
            return f"server is up but has no model {self.model!r}; available: {', '.join(names) or 'none'}"
        return ""


@dataclass
class SilentStub:
    """Reports nothing. The floor: perfect precision, zero recall."""

    name: str = "stub:silent"

    def complete(self, system: str, user: str, schema: dict) -> Response:
        return Response(text=contract.render([]))


STUBS = {"silent": SilentStub}
