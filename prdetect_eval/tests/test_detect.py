"""Tests for the detector path: packing, the response contract, the stubs.

The scoring harness has its own tests. These cover the failures that would look
exactly like a bad model: a code block whose line numbers do not match the file,
a prompt that drifts between calls and quietly disables prefix caching, or a
parser that discards malformed answers instead of counting them.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates as candidate_module
import run_detect
from adapters import load_cases
from candidates import authz as authz_module
from candidates.context import Tree
from detect import client as client_module
from detect import context_pack, contract, prompt
from schema import Case


@pytest.fixture(scope="session")
def cases() -> list[Case]:
    return load_cases()


@pytest.fixture(scope="session")
def packs(cases) -> list[context_pack.Pack]:
    return run_detect.build_packs(cases, ["authz"])


def test_every_case_is_asked(cases, packs):
    """A type whose defect is an absence must question clean code too."""
    assert {pack.case_id for pack in packs} == {case.case_id for case in cases}


def test_prompt_prefix_is_byte_identical(packs):
    """Prefix caching is the load-bearing optimization; drift here is silent."""
    assert len({pack.system for pack in packs}) == 1
    assert prompt.SYSTEM == prompt.system()
    assert "{" not in prompt.SYSTEM.split("## How to answer")[0]


def test_code_block_line_numbers_match_the_file(cases, packs):
    by_id = {case.case_id: case for case in cases}
    for pack in packs:
        case = by_id[pack.case_id]
        source = case.head_files.get(pack.filename)
        if source is None:
            continue
        lines = source.split("\n")
        for text in pack.user.split("\n"):
            head, _, body = text.partition(" | ")
            number = head[:5].strip()
            if not number.isdigit():
                continue
            assert lines[int(number) - 1] == body, f"{pack.case_id} {pack.filename}:{number}"


def test_changed_lines_are_marked(cases, packs):
    by_id = {case.case_id: case for case in cases}
    marked = 0
    for pack in packs:
        added = by_id[pack.case_id].added_lines.get(pack.filename, frozenset())
        for text in pack.user.split("\n"):
            head, _, _ = text.partition(" | ")
            number, mark = head[:5].strip(), head[5:].strip()
            if not number.isdigit():
                continue
            assert (mark == "+") == (int(number) in added)
            marked += mark == "+"
    assert marked > 0


def test_every_flagged_site_appears_in_its_code_block(packs):
    for pack in packs:
        shown = {
            int(text[:5]) for text in pack.user.split("\n")
            if text[:5].strip().isdigit() and " | " in text
        }
        assert set(pack.focus_lines) <= shown, pack.case_id


def test_decorators_are_shown_with_their_function(cases):
    """``ast`` starts a function at ``def``, below ``@permission_classes(...)``."""
    source = "\n".join([
        "import x",
        "",
        "@permission_classes([IsOrgMember])",
        "def handler(request):",
        "    return X.objects.all()",
    ])
    case = _synthetic_case(source, added={4, 5})
    tree = Tree(case)
    found = candidate_module.enumerate_case(case, ["authz"])
    assert found, "expected an ORM read candidate"
    pack = context_pack.build(case, tree, found, "apps/demo/views.py")
    assert "@permission_classes([IsOrgMember])" in pack.user


def _synthetic_case(source: str, added: set[int]) -> Case:
    name = "apps/demo/views.py"
    return Case(
        case_id="synthetic", pair_id=None, variant="buggy", difficulty="easy",
        primary_type="authz", is_defective=True, pr_title="t", pr_description="d",
        changed_files=(name,), noise_files=(), deleted_files=(),
        added_lines={name: frozenset(added)}, head_files={name: source}, diff="",
        labels=(), distractors=(),
    )


@pytest.mark.parametrize("code,expected", [
    ("X.objects.all().select_related('organization')", False),
    ("X.objects.filter(organization_id=self.request.user.organization_id)", True),
    ("org_rows(self.request.user.organization_id)", True),
    ("X.objects.filter(id=pk)", False),
    ("X.objects.filter(Q(organization=org))", True),
])
def test_scoping_predicate(code, expected):
    """``select_related('organization')`` is a join hint, not tenant scoping."""
    import ast

    assert authz_module.narrows_to_organization(ast.parse(code)) is expected


def test_parse_reads_a_valid_response():
    reports, rejects = contract.parse(json.dumps({"findings": [
        {"file": "a/b.py", "line": 12, "quote": "q", "reason": "r", "confidence": 0.7}
    ]}))
    assert not rejects
    assert reports == [contract.Report("a/b.py", 12, "q", "r", 0.7)]


def test_parse_accepts_an_empty_verdict():
    reports, rejects = contract.parse('{"findings": []}')
    assert (reports, rejects) == ([], [])


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
        {"file": "a.py", "line": 1, "quote": "", "reason": "", "confidence": 9.0}
    ]}))
    assert reports[0].confidence == 1.0


def test_silent_stub_reports_nothing(packs):
    reports, rejects = contract.parse(
        client_module.SilentStub().complete(packs[0], contract.RESPONSE_SCHEMA).text
    )
    assert (reports, rejects) == ([], [])


def test_flag_all_stub_returns_every_site(packs):
    """The stub is the ceiling re-derived through packing, parsing and mapping."""
    stub = client_module.FlagAllStub()
    total = 0
    for pack in packs:
        reports, rejects = contract.parse(stub.complete(pack, contract.RESPONSE_SCHEMA).text)
        assert not rejects
        assert [report.line for report in reports] == [c.focus.start_line for c in pack.candidates]
        assert all(report.quote for report in reports)
        total += len(reports)
    assert total == sum(len(pack.candidates) for pack in packs)


def test_predictions_carry_the_detector_type(packs):
    stub = client_module.FlagAllStub()
    reports, _ = contract.parse(stub.complete(packs[0], contract.RESPONSE_SCHEMA).text)
    predictions = run_detect.to_predictions(packs[0], reports, "authz")
    assert all(p.type == "authz" and p.case_id == packs[0].case_id for p in predictions)
    assert all(p.detector == run_detect.DETECTOR and p.stage == "detect" for p in predictions)


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
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        _FakeOllama.seen = payload
        self._send({
            "message": {"content": contract.render(
                [contract.Report("apps/demo/views.py", 5, "q", "r", 0.9)]
            )},
            "prompt_eval_count": 1234, "eval_count": 20, "done": True,
        })


@pytest.fixture(scope="module")
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_client_speaks_the_ollama_api(fake_server, packs):
    client = client_module.OllamaClient(model="test-model", base_url=fake_server)
    assert client.health() == ""

    response = client.complete(packs[0], contract.RESPONSE_SCHEMA)
    sent = _FakeOllama.seen
    assert sent["model"] == "test-model"
    assert sent["stream"] is False
    assert sent["format"] == contract.RESPONSE_SCHEMA
    assert [message["role"] for message in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == prompt.SYSTEM
    assert sent["options"]["temperature"] == 0.0

    assert response.prompt_tokens == 1234
    reports, rejects = contract.parse(response.text)
    assert not rejects and reports[0].line == 5


def test_client_reports_a_missing_model(fake_server):
    assert "no model" in client_module.OllamaClient(model="absent", base_url=fake_server).health()


def test_client_survives_an_unreachable_server(packs):
    client = client_module.OllamaClient(
        model="m", base_url="http://127.0.0.1:9", timeout=1.0, retries=0
    )
    response = client.complete(packs[0], contract.RESPONSE_SCHEMA)
    assert response.error and response.text == ""
    assert contract.parse(response.text)[1], "an unreachable server must be recorded, not silent"


def test_response_schema_is_a_closed_contract():
    item = contract.RESPONSE_SCHEMA["properties"]["findings"]["items"]
    assert set(item["required"]) == set(item["properties"])
    assert contract.RESPONSE_SCHEMA["required"] == ["findings"]
