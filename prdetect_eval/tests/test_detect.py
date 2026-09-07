"""Tests for the detector path: packing, the response contract, the client.

These cover the failures that would look exactly like a bad model: a code block
whose line numbers do not match the file, a prompt that drifts between calls and
quietly disables prefix caching, or a parser that discards malformed answers
instead of counting them.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters import load_cases
from detect import client as client_module
from detect import contract, pack, prompt

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


@pytest.fixture(scope="session", params=["demo_repo", "swrbench"])
def dataset(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def packs(dataset) -> list[pack.Pack]:
    cases = load_cases(DATASETS / f"{dataset}.eval.jsonl")
    return [pack.build(case, dataset) for case in cases]


def test_every_case_is_asked(dataset, packs):
    cases = load_cases(DATASETS / f"{dataset}.eval.jsonl")
    assert {p.case_id for p in packs} == {c.case_id for c in cases}


def test_prompt_prefix_is_byte_identical(packs):
    """Prefix caching is the load-bearing optimization; drift here is silent."""
    assert len({p.system for p in packs}) == 1


def test_prompt_lists_exactly_the_dataset_types(dataset):
    text = prompt.system(dataset)
    for name in prompt.types(dataset):
        assert f"`{name}`" in text, name
    other = "swrbench" if dataset == "demo_repo" else "demo_repo"
    for name in prompt.types(other):
        assert f"`{name}`" not in text, f"{name} leaked into {dataset}"


def test_code_block_line_numbers_match_the_file(dataset):
    """The model answers with file:line, so renumbering would make every hit wrong."""
    if dataset != "demo_repo":
        pytest.skip("SWRBench ships hunks, not whole files")
    for case in load_cases(DATASETS / "demo_repo.eval.jsonl"):
        item = pack.build(case, "demo_repo")
        current = None
        for text in item.user.split("\n"):
            if text.startswith("# FILE "):
                current = case.head_files[text[len("# FILE "):].strip()].split("\n")
                continue
            head, _, body = text.partition(" | ")
            number = head[:5].strip()
            if current is None or not number.isdigit():
                continue
            assert current[int(number) - 1] == body, f"{case.case_id}:{number}"


def test_changed_lines_are_marked(dataset):
    if dataset != "demo_repo":
        pytest.skip("SWRBench ships hunks, not whole files")
    marked = 0
    for case in load_cases(DATASETS / "demo_repo.eval.jsonl"):
        item = pack.build(case, "demo_repo")
        current = None
        for text in item.user.split("\n"):
            if text.startswith("# FILE "):
                current = case.added_lines.get(text[len("# FILE "):].strip(), frozenset())
                continue
            head, _, _ = text.partition(" | ")
            number, mark = head[:5].strip(), head[5:].strip()
            if current is None or not number.isdigit():
                continue
            assert (mark == "+") == (int(number) in current)
            marked += mark == "+"
    assert marked > 0


def test_cross_file_cases_show_every_changed_file():
    """Eight demo_repo defects only exist between two files; splitting hides them."""
    for case in load_cases(DATASETS / "demo_repo.eval.jsonl"):
        if not any(label.cross_file for label in case.labels):
            continue
        text = pack.build(case, "demo_repo").user
        for filename in case.head_files:
            assert f"# FILE {filename}" in text, f"{case.case_id} hides {filename}"


def test_schema_constrains_the_type_to_the_taxonomy(dataset):
    schema = contract.response_schema(prompt.types(dataset))
    assert schema["properties"]["findings"]["items"]["properties"]["type"]["enum"] == prompt.types(dataset)


def test_parse_reads_a_valid_response():
    reports, rejects = contract.parse(json.dumps({"findings": [
        {"file": "a/b.py", "line": 12, "type": "authz", "title": "t", "confidence": 0.7}
    ]}))
    assert not rejects
    assert reports == [contract.Report("a/b.py", 12, "authz", "t", 0.7)]


def test_parse_accepts_an_empty_verdict():
    assert contract.parse('{"findings": []}') == ([], [])


@pytest.mark.parametrize("payload", ["", "not json", "[]", '{"other": 1}', '{"findings": {}}'])
def test_parse_rejects_malformed_responses(payload):
    reports, rejects = contract.parse(payload)
    assert not reports and len(rejects) == 1


@pytest.mark.parametrize("finding", [
    {"file": "a.py", "line": 0}, {"file": "a.py", "line": "x"}, {"file": "", "line": 3}, {"line": 3},
])
def test_parse_rejects_unusable_findings(finding):
    """A location that cannot exist is counted, not silently dropped."""
    reports, rejects = contract.parse(json.dumps({"findings": [finding]}))
    assert not reports and len(rejects) == 1


def test_confidence_is_clamped():
    reports, _ = contract.parse(json.dumps({"findings": [
        {"file": "a.py", "line": 1, "type": "authz", "title": "", "confidence": 9.0}
    ]}))
    assert reports[0].confidence == 1.0


def test_silent_stub_reports_nothing():
    response = client_module.SilentStub().complete("s", "u", contract.response_schema(["authz"]))
    assert contract.parse(response.text) == ([], [])


class _FakeOllama(BaseHTTPRequestHandler):
    """Enough of the Ollama API to prove the client speaks it correctly."""

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
        self._send({"models": [{"name": "test-model:latest"}]})

    def do_POST(self) -> None:
        _FakeOllama.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        self._send({
            "message": {"content": contract.render(
                [contract.Report("app/orders.py", 5, "authz", "t", 0.9)]
            )},
            "prompt_eval_count": 1234, "eval_count": 20, "done": True,
        })


@pytest.fixture(scope="module")
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_client_speaks_the_ollama_api(fake_server):
    schema = contract.response_schema(["authz"])
    client = client_module.OllamaClient(model="test-model", base_url=fake_server)
    assert client.health() == ""

    response = client.complete("SYSTEM", "USER", schema)
    sent = _FakeOllama.seen
    assert sent["model"] == "test-model" and sent["stream"] is False
    assert sent["format"] == schema
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "SYSTEM"
    assert sent["options"]["temperature"] == 0.0
    assert response.prompt_tokens == 1234
    reports, rejects = contract.parse(response.text)
    assert not rejects and reports[0].line == 5


def test_client_reports_a_missing_model(fake_server):
    assert "no model" in client_module.OllamaClient(model="absent", base_url=fake_server).health()


def test_client_survives_an_unreachable_server():
    client = client_module.OllamaClient(model="m", base_url="http://127.0.0.1:9", timeout=1.0, retries=0)
    response = client.complete("s", "u", contract.response_schema(["authz"]))
    assert response.error and response.text == ""
    assert contract.parse(response.text)[1], "an unreachable server must be recorded, not silent"
