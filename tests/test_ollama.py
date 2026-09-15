"""The Ollama client, against a fake server that speaks enough of the API."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from prdetect.detect import contract, ollama


class FakeOllama(BaseHTTPRequestHandler):
    seen: dict = {}

    def log_message(self, *args) -> None:
        pass

    def _send(self, document: dict) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"models": [{"name": "test-model:latest", "digest": "deadbeef" * 8,
                                "details": {"quantization_level": "Q4_K_M"}}]})

    def do_POST(self) -> None:
        FakeOllama.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        self._send({"message": {"content": contract.render([contract.Report("app/orders.py", 5, "xss", "t", 0.9)])},
                    "prompt_eval_count": 1234, "eval_count": 20, "done": True})


class DyingOllama(FakeOllama):
    """Healthy enough to start, gone by the first call -- a dropped tunnel."""

    def do_POST(self) -> None:
        self.send_error(502, "tunnel gone")


def serve(handler) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture(scope="module")
def fake_server():
    server, url = serve(FakeOllama)
    yield url
    server.shutdown()


def test_the_silent_stub_reports_nothing():
    response = ollama.SilentStub().complete("s", "u", contract.response_schema(["xss"]))
    assert contract.parse(response.text) == ([], [])


def test_the_client_speaks_the_ollama_api(fake_server):
    schema = contract.response_schema(["xss"])
    client = ollama.OllamaClient(model="test-model", base_url=fake_server)
    assert client.health() == ""
    response = client.complete("SYSTEM", "USER", schema)
    sent = FakeOllama.seen
    assert sent["model"] == "test-model" and sent["stream"] is False
    assert sent["format"] == schema
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "SYSTEM"
    assert sent["options"]["temperature"] == 0.0
    assert response.prompt_tokens == 1234 and response.thinking == ""
    reports, rejects = contract.parse(response.text)
    assert not rejects and reports[0].line == 5


def test_the_client_reports_the_served_build(fake_server):
    assert ollama.OllamaClient(model="test-model", base_url=fake_server).build().startswith("deadbeef")
    assert ollama.OllamaClient(model="absent", base_url=fake_server).build() == ""
    assert ollama.OllamaClient(model="m", base_url="http://127.0.0.1:9", timeout=1).build() == ""


def test_the_client_reports_a_missing_model(fake_server):
    assert "no model" in ollama.OllamaClient(model="absent", base_url=fake_server).health()


def test_the_client_survives_an_unreachable_server():
    client = ollama.OllamaClient(model="m", base_url="http://127.0.0.1:9", timeout=1.0, retries=0)
    response = client.complete("s", "u", contract.response_schema(["xss"]))
    assert response.error and response.text == ""
    assert contract.parse(response.text)[1], "an unreachable server is recorded, not silent"


def test_the_client_keeps_the_thinking_it_is_given():
    class Thinker(FakeOllama):
        def do_POST(self) -> None:
            self._send({"message": {"content": contract.render([]), "thinking": "step one"},
                        "prompt_eval_count": 5, "done": True})

    server, url = serve(Thinker)
    response = ollama.OllamaClient(model="test-model", base_url=url).complete("s", "u", {})
    server.shutdown()
    assert response.thinking == "step one"
    assert contract.parse(response.text) == ([], [])
