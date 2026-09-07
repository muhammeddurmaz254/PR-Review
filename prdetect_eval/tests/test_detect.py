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
from statistics import median
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


def test_measured_prompts_are_unchanged():
    """Each number in the plan was measured under one exact text. A silent edit
    invalidates it without anything failing, so every version that has a
    published score is pinned here and unpinned only by re-measuring."""
    import hashlib
    pinned = {
        ("demo_repo", "review/v1"): "4b1de85b72188b80",
        ("swrbench", "review/v1"): "9be10ddda5fefa3f",
    }
    for (dataset, version), digest in pinned.items():
        actual = hashlib.sha256(prompt.system(dataset, version).encode("utf-8")).hexdigest()
        assert actual.startswith(digest), f"{dataset} {version} changed; re-measure or revert"


def test_every_prompt_version_renders_for_every_dataset():
    for version in prompt.VERSIONS:
        for dataset in prompt.TAXONOMIES:
            text = prompt.system(dataset, version)
            assert "{format}" not in text and text.strip()


def test_the_type_names_never_move_between_versions():
    """The schema enum and every stored label are built from these names."""
    for version in prompt.VERSIONS:
        _, taxonomies = prompt.VERSIONS[version]
        for dataset in prompt.TAXONOMIES:
            assert list(taxonomies[dataset]) == prompt.types(dataset), (version, dataset)


def test_v3_separates_the_two_definitions_that_collided():
    """v1 filed 'a dependency is unavailable' under F.3 and the corpus files it
    under F.5, which is a large part of why F.3 was never once reported."""
    v1 = prompt.SWRBENCH_TYPES
    v3 = prompt.SWRBENCH_TYPES_V3
    assert "dependency is unavailable" in v1["F.3 Resource"]
    assert "dependency" not in v3["F.3 Resource"]
    assert "dependency" in v3["F.5 Support"]


def test_v3_covers_the_shapes_v1_left_out(dataset):
    """Two of F.1's five cases are a deleted public name, and two of F.4's are a
    guard that exists and is wrong. v1 described neither."""
    text = prompt.system("swrbench", "review/v3")
    for phrase in ("removed or narrowed", "asserts something that",
                   "not on its sibling", "never imported"):
        assert phrase in text, phrase


def test_v3_leaves_demo_repo_alone():
    _, v1 = prompt.VERSIONS["review/v1"]
    _, v3 = prompt.VERSIONS["review/v3"]
    assert v1["demo_repo"] == v3["demo_repo"]


def test_v2_moves_the_trade_off_out_of_the_model():
    """v1 asserted the prior and the cost; v2 grades the doubt instead."""
    v1 = prompt.system("swrbench", "review/v1")
    v2 = prompt.system("swrbench", "review/v2")
    assert "around 0.3" not in v1
    assert "a miss costs one line of recall" in v1
    assert "a miss costs one line of recall" not in v2
    assert "around 0.3" in v2, "v2 must anchor the confidence scale"


def test_each_prompt_describes_its_own_print_format(dataset):
    """Explaining the wrong rendering is worse than explaining none."""
    text = prompt.system(dataset)
    if dataset == "swrbench":
        assert "`-` and no number" in text and "# FILE" in text
        assert "the code it changed with real line numbers" not in text
    else:
        assert "Lines marked `+`" in text and "`-` and no number" not in text


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


SAMPLE_DIFF = """\
# commit 6c2f673daf8d Have same name for fulltrace
src/_pytest/terminal.py
@@ -114,7 +114,7 @@ def pytest_addoption(parser):
     )
     group._addoption(
-        "--full-trace",
+        "--fulltrace",
         action="store_true",

# commit 958374ad0000 Remove name from author
AUTHORS
@@ -20,3 +20,2 @@
 Anthony Sottile
-Someone Else
 Zac Hatfield-Dodds
"""


def test_parse_diff_renumbers_onto_the_file():
    first, second = pack.parse_diff(SAMPLE_DIFF)
    assert (first.filename, first.start, first.end) == ("src/_pytest/terminal.py", 114, 117)
    assert first.commit.startswith("6c2f673")
    assert first.heading == "def pytest_addoption(parser):"
    assert second.filename == "AUTHORS"


def test_parse_diff_gives_a_removed_line_no_number():
    """A deleted line is not in the new file, but a defect can be the deletion."""
    removed = [row for row in pack.parse_diff(SAMPLE_DIFF)[0].lines if row[5:8] == " - "]
    assert len(removed) == 1
    assert removed[0].startswith("      - | ") and "--full-trace" in removed[0]


def test_swrbench_line_numbers_follow_the_hunk_header(dataset):
    """The label is a file line, so a hunk that renumbers is a guaranteed miss."""
    if dataset != "swrbench":
        pytest.skip("demo_repo prints whole files")
    for case in load_cases(DATASETS / "swrbench.eval.jsonl"):
        for hunk in pack.parse_diff(case.diff):
            expected = hunk.start
            for row in hunk.lines:
                head = row[:5].strip()
                if not head:
                    continue
                assert int(head) == expected, f"{case.case_id} {hunk.filename}"
                expected += 1


def test_swrbench_marks_exactly_the_added_lines(dataset):
    if dataset != "swrbench":
        pytest.skip("demo_repo prints whole files")
    for case in load_cases(DATASETS / "swrbench.eval.jsonl"):
        marked: dict[str, set[int]] = {}
        for hunk in pack.parse_diff(case.diff):
            for row in hunk.lines:
                if row[5:8] == " + ":
                    marked.setdefault(hunk.filename, set()).add(int(row[:5]))
        assert marked == {name: set(lines) for name, lines in case.added_lines.items()}, case.case_id


def test_every_pack_shows_some_code(dataset, packs):
    """All twenty-five clean SWRBench cases once had an empty diff and an empty
    file list, so their zero false alarms measured an empty prompt rather than a
    model. Nothing here may be asked about code it was not shown."""
    for item in packs:
        assert item.shown_lines > 0, item.case_id


def test_clean_and_defective_packs_are_the_same_size(dataset, packs):
    """Length must not be a shortcut to the verdict."""
    cases = {case.case_id: case for case in load_cases(DATASETS / f"{dataset}.eval.jsonl")}
    defective = sorted(p.shown_lines for p in packs if cases[p.case_id].is_defective)
    clean = sorted(p.shown_lines for p in packs if not cases[p.case_id].is_defective)
    if not clean or not defective:
        pytest.skip("one-sided dataset")
    ratio = median(defective) / median(clean)
    assert 0.5 < ratio < 2.0, f"{dataset}: {median(defective)} vs {median(clean)} lines"


def test_every_label_is_printed_in_its_pack(dataset, packs):
    """A finding the pack never shows is unreachable, not hard."""
    by_id = {item.case_id: item for item in packs}
    for case in load_cases(DATASETS / f"{dataset}.eval.jsonl"):
        text = by_id[case.case_id].user
        current = ""
        for label in case.labels:
            found = False
            for row in text.split("\n"):
                if row.startswith("# FILE "):
                    current = row[len("# FILE "):].split("   (commit")[0].strip()
                    continue
                head = row[:5].strip()
                if current == label.span.file and head.isdigit():
                    if label.span.start_line <= int(head) <= label.span.end_line:
                        found = True
                        break
            assert found, f"{case.case_id}: {label.span.file}:{label.span.start_line} not shown"


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
