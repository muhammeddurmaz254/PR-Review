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
from detect import anchor, challenge, contract, pack, prompt

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


@pytest.fixture(scope="session", params=["demo_repo", "swrbench", "halka"])
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
        ("swrbench", "review/v3"): "243f3e371087abb8",
        ("swrbench", "review/v4"): "3abe68a78ff83ab3",
        ("swrbench", "review/v5"): "368c076c6929b3e9",
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
        self._send({"models": [{
            "name": "test-model:latest", "digest": "deadbeef" * 8,
            "details": {"quantization_level": "Q4_K_M"},
        }]})

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


def test_the_client_reports_the_served_build(fake_server):
    """A resumed run can be stitched from two machines; the digest is what makes
    "same model" checkable instead of assumed."""
    client = client_module.OllamaClient(model="test-model", base_url=fake_server)
    assert client.build().startswith("deadbeef")
    assert client_module.OllamaClient(model="absent", base_url=fake_server).build() == ""
    assert client_module.OllamaClient(model="m", base_url="http://127.0.0.1:9", timeout=1).build() == ""


def test_client_reports_a_missing_model(fake_server):
    assert "no model" in client_module.OllamaClient(model="absent", base_url=fake_server).health()


def test_client_survives_an_unreachable_server():
    client = client_module.OllamaClient(model="m", base_url="http://127.0.0.1:9", timeout=1.0, retries=0)
    response = client.complete("s", "u", contract.response_schema(["authz"]))
    assert response.error and response.text == ""
    assert contract.parse(response.text)[1], "an unreachable server must be recorded, not silent"


CHALLENGE_ROWS = [
    "  115 + |         np.clip(candidate_ids, None, len(closest_dist_sq) - 1,",
    "  116 + |                 out=candidate_ids)",
]


def test_an_unreadable_verdict_leaves_the_claim_standing():
    """A stage that drops findings when the server hiccups reports a precision
    gain it did not earn."""
    for payload in ("", "not json", "[]", '{"verdict": "maybe"}'):
        assert challenge.parse(payload)[0] == "stands"


def test_refuting_requires_quoting_the_excerpt():
    """Otherwise the stage can refute anything by asserting a contradiction that
    is not there -- the same failure it exists to catch."""
    assert challenge.honours("out=candidate_ids", CHALLENGE_ROWS)
    assert not challenge.honours("the code is clearly fine", CHALLENGE_ROWS)
    assert not challenge.honours("", CHALLENGE_ROWS)


def test_the_quote_check_ignores_gutters_and_escaping():
    """Both mismatches were measured, and both rejected a correct refutation."""
    assert challenge.honours("  116 + |                 out=candidate_ids)", CHALLENGE_ROWS)
    assert challenge.honours('dquotes=("\\sphinxquotedblleft{}",',
                             ['  2213 + |     dquotes=("\\\\sphinxquotedblleft{}",'])


def test_the_challenge_prompt_defaults_to_keeping_the_finding():
    text = challenge.SYSTEM
    assert '"stands" is the answer' in text
    assert "Quote the exact text" in text


def test_the_excerpt_carries_the_detector_line_numbers(dataset):
    """The claim names a line; an excerpt renumbered against it proves nothing."""
    cases = load_cases(DATASETS / f"{dataset}.eval.jsonl")
    checked = 0
    for case in cases:
        for label in case.labels:
            rows = challenge.excerpt(case, label.span.file, label.span.start_line)
            numbers = [int(row[:5]) for row in rows if row[:5].strip()]
            if not numbers:
                continue
            checked += 1
            assert min(numbers) <= label.span.start_line <= max(numbers), case.case_id
    assert checked


def _case(dataset: str, case_id: str = ""):
    cases = load_cases(DATASETS / f"{dataset}.eval.jsonl")
    return next((c for c in cases if c.case_id == case_id), cases[0])


def test_shown_lines_are_exactly_what_the_pack_printed(dataset):
    """The filter compares against these; if they drift it rejects real findings."""
    for case in load_cases(DATASETS / f"{dataset}.eval.jsonl"):
        numbered, _ = pack.shown_lines(case)
        text = pack.build(case, dataset).user
        current = ""
        for row in text.split("\n"):
            if row.startswith("# FILE "):
                current = row[len("# FILE "):].split("   (commit")[0].strip()
                continue
            head, _, body = row.partition("| ")
            number = head[:5].strip()
            if current and number.isdigit() and int(number) in numbered.get(current, {}):
                assert body in numbered[current][int(number)], f"{case.case_id}:{number}"


def test_a_quote_that_matches_its_line_is_anchored(dataset):
    case = _case(dataset)
    numbered, _ = pack.shown_lines(case)
    filename = sorted(numbered)[0]
    line, texts = next((n, t) for n, t in sorted(numbered[filename].items())
                       if len(t[0].strip()) > 12)
    report = contract.Report(filename, line, "x", "t", 0.9, quote=texts[0])
    assert anchor.resolve([report], case)[0].verdict == "anchored"


def test_a_quote_from_elsewhere_snaps_to_where_it_actually_is(dataset):
    """The claim is about real code and only the number is wrong, so correcting
    the number is localization the detector did not have to earn twice."""
    case = _case(dataset)
    numbered, _ = pack.shown_lines(case)
    filename = sorted(numbered)[0]
    rows = [(n, t[0]) for n, t in sorted(numbered[filename].items()) if len(t[0].strip()) > 12]
    unique = [(n, t) for n, t in rows
              if sum(1 for _, other in rows if anchor.normalise(t) in anchor.normalise(other)) == 1]
    if not unique:
        pytest.skip("no line unique enough in this case")
    line, text = unique[0]
    wrong = line + 500
    decision = anchor.resolve([contract.Report(filename, wrong, "x", "t", 0.9, quote=text)], case)[0]
    assert decision.verdict == "snapped" and decision.line == line
    assert anchor.apply([decision])[0].line == line


def test_a_quote_that_was_never_shown_is_dropped(dataset):
    case = _case(dataset)
    numbered, _ = pack.shown_lines(case)
    filename = sorted(numbered)[0]
    report = contract.Report(filename, 1, "x", "t", 0.9,
                             quote="raise NotImplementedError('nothing prints this')")
    decision = anchor.resolve([report], case)[0]
    assert decision.verdict == "unsupported" and not decision.kept
    assert anchor.apply([decision]) == []


def test_a_deleted_line_can_be_quoted_but_not_anchored():
    """A defect that *is* the removal has no numbered line to point at."""
    case = _case("swrbench", "pytest-dev__pytest-5668")
    _, deleted = pack.shown_lines(case)
    filename = next(iter(deleted))
    decision = anchor.resolve(
        [contract.Report(filename, 117, "x", "t", 0.9, quote=deleted[filename][0])], case)[0]
    assert decision.verdict in {"deleted-line", "anchored"} and decision.kept


def test_an_absent_quote_is_never_used_to_drop_a_finding(dataset):
    """Versions before v4 do not ask for one; the filter must stay out of the way."""
    case = _case(dataset)
    decisions = anchor.resolve([contract.Report("a.py", 1, "x", "t", 0.9)], case)
    assert decisions[0].verdict == "unchecked" and decisions[0].kept


def test_v4_asks_for_the_quote_and_the_schema_requires_it():
    assert "review/v4" in prompt.QUOTED
    assert "`quote` is the code on that line" in prompt.system("swrbench", "review/v4")
    assert "quote" not in prompt.system("swrbench", "review/v3")
    schema = contract.response_schema(["F.2 Logic"], quote=True)
    assert "quote" in schema["properties"]["findings"]["items"]["required"]
    plain = contract.response_schema(["F.2 Logic"])
    assert "quote" not in plain["properties"]["findings"]["items"]["properties"]


def test_the_cap_keeps_the_most_confident_findings():
    reports = [contract.Report("a.py", n, "x", "t", c)
               for n, c in ((1, 0.2), (2, 0.9), (3, 0.5), (4, 0.7))]
    kept = contract.cap(reports, 2)
    assert [r.line for r in kept] == [2, 4]
    assert contract.cap(reports, 0) == reports
    assert len(contract.cap(reports, 10)) == 4


def test_the_cap_sits_above_the_corpus_label_density():
    """A cap at the density would be tuning against the answer key."""
    for dataset in ("demo_repo", "swrbench"):
        cases = load_cases(DATASETS / f"{dataset}.eval.jsonl")
        densest = max(len(case.scored_labels) for case in cases)
        assert densest <= 3, f"{dataset} carries {densest} required labels on one case"


def test_splitting_is_off_by_default_in_build(dataset):
    """`build` is the whole pull request; only `split` with a limit divides it."""
    for case in load_cases(DATASETS / f"{dataset}.eval.jsonl"):
        assert pack.build(case, dataset).parts == 1


def test_a_split_keeps_every_line_exactly_once(dataset):
    """A hunk dropped or duplicated between excerpts is a silent recall loss."""
    for case in load_cases(DATASETS / f"{dataset}.eval.jsonl"):
        whole = pack.build(case, dataset)
        parts = pack.split(case, dataset, max_lines=120)
        assert sum(p.shown_lines for p in parts) == whole.shown_lines, case.case_id
        assert {p.part for p in parts} == set(range(1, len(parts) + 1))


def test_a_split_never_divides_a_hunk(dataset):
    if dataset != "swrbench":
        pytest.skip("demo_repo prints whole files and is never split")
    for case in load_cases(DATASETS / "swrbench.eval.jsonl"):
        parts = pack.split(case, "swrbench", max_lines=120)
        seen = [line for p in parts for line in p.user.split("\n") if line.startswith("Lines ")]
        whole = [line for line in pack.build(case, "swrbench").user.split("\n")
                 if line.startswith("Lines ")]
        assert seen == whole, case.case_id


def test_whole_file_cases_are_never_split():
    """Eight demo_repo defects exist only between two changed files; separating
    them would make those unreachable however small the limit."""
    for case in load_cases(DATASETS / "demo_repo.eval.jsonl"):
        assert len(pack.split(case, "demo_repo", max_lines=1)) == 1


def test_a_split_pack_says_which_excerpt_it_is(dataset):
    if dataset != "swrbench":
        pytest.skip("demo_repo is never split")
    divided = [p for case in load_cases(DATASETS / "swrbench.eval.jsonl")
               for p in pack.split(case, "swrbench", max_lines=120) if p.parts > 1]
    assert divided
    for item in divided:
        assert f"excerpt {item.part} of {item.parts}" in item.user
    assert len({p.system for p in divided}) == 1, "the cached prefix must not carry the part"


def test_every_label_is_reachable_after_a_split(dataset):
    """The split must not put a finding in no excerpt at all."""
    for case in load_cases(DATASETS / f"{dataset}.eval.jsonl"):
        text = "\n".join(p.user for p in pack.split(case, dataset, max_lines=120))
        for label in case.labels:
            current = ""
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
            assert found, f"{case.case_id}: {label.span.file}:{label.span.start_line}"


def test_v5_puts_support_code_in_scope():
    """Six of the seven findings the model stayed silent on live in a test, a
    fixture, an example or a packaging script."""
    v4 = prompt.system("swrbench", "review/v4")
    v5 = prompt.system("swrbench", "review/v5")
    assert "A test is not exempt for being a test" in v5
    assert "A test is not exempt" not in v4
    for shape in ("skip condition that is always true", "conftest", "setup.py"):
        assert shape in v5, shape


def test_v5_narrows_the_two_clauses_that_silenced_it():
    """`a placeholder in an example file` reads as the file, not the placeholder,
    and `a broad except` is the whole of one F.4 label."""
    v5 = prompt.system("swrbench", "review/v5")
    assert "a placeholder in an example file" not in v5
    assert "about the expression, not the file it is in" in v5
    assert "swallows rather than re-raises is a defect" in v5


def test_v5_still_asks_for_the_quote():
    assert "review/v5" in prompt.QUOTED
    assert "`quote` is the code on that line" in prompt.system("swrbench", "review/v5")


def test_halka_carries_the_repository_behind_the_diff():
    """Thirty-three of forty-nine defects are only legible against code the pull
    request does not touch; a pack without it cannot reach them."""
    cases = load_cases(DATASETS / "halka.eval.jsonl")
    assert all(case.context_files for case in cases)
    for case in cases:
        assert not (set(case.context_files) & set(case.head_files)), case.case_id


def test_with_repo_adds_the_unchanged_files_and_nothing_else():
    case = load_cases(DATASETS / "halka.eval.jsonl")[0]
    lean, full = pack.build(case, "halka"), pack.build(case, "halka", with_repo=True)
    assert lean.shown_lines == full.shown_lines, "changed-code accounting must not move"
    assert full.estimated_tokens > lean.estimated_tokens * 3
    assert lean.user in full.user or "# WHAT THIS PULL REQUEST CHANGED" in full.user
    for filename in case.context_files:
        assert f"# FILE {filename}   (unchanged)" in full.user
        assert filename not in lean.user


def test_the_anchor_filter_sees_the_repository_when_the_pack_does():
    """Otherwise every quote taken from an unchanged file is called invented."""
    case = load_cases(DATASETS / "halka.eval.jsonl")[0]
    filename, line, text = next(
        (name, n, row)
        for name in sorted(case.context_files)
        for n, row in enumerate(case.context_files[name].split("\n"), 1)
        if len(row.strip()) > 24
    )
    report = contract.Report(filename, line, "sql_injection", "t", 0.9, quote=text)
    assert anchor.resolve([report], case)[0].verdict == "unchecked"
    assert anchor.resolve([report], case, with_repo=True)[0].verdict == "anchored"


def test_every_halka_type_is_defined_once():
    assert len(prompt.HALKA_TYPES) == 41
    text = prompt.system("halka", "review/v5")
    for name in prompt.HALKA_TYPES:
        assert f"`{name}`" in text, name


def test_resume_refuses_to_mix_two_prompts(tmp_path):
    """A run stitched from two prompt versions is worse than no run."""
    import run_detect
    run_dir = tmp_path / "half"
    run_dir.mkdir()
    (run_dir / "system_prompt.txt").write_text("a prompt this run was not made under\n")
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"case_id": "x", "part": 1, "text": '{"findings": []}'}) + "\n")
    with pytest.raises(SystemExit) as raised:
        run_detect.main([
            "--dataset", "halka", "--stub", "silent", "--resume",
            "--run-id", "half", "--out", str(tmp_path), "--quiet",
        ])
    assert "different system prompt" in str(raised.value)


def test_resume_reuses_the_answers_already_on_disk(tmp_path):
    import run_detect
    args = ["--dataset", "halka", "--stub", "silent", "--limit", "4",
            "--run-id", "part", "--out", str(tmp_path), "--quiet"]
    run_detect.main(args)
    first = (tmp_path / "part" / "responses.jsonl").read_text().splitlines()
    assert len(first) == 4

    # Truncate to two answers, then resume: the two survivors are not re-asked.
    (tmp_path / "part" / "responses.jsonl").write_text("\n".join(first[:2]) + "\n")
    run_detect.main(args + ["--resume"])
    manifest = json.loads((tmp_path / "part" / "config.json").read_text())
    assert manifest["reused_answers"] == 2
    assert len((tmp_path / "part" / "responses.jsonl").read_text().splitlines()) == 4


class _DyingOllama(_FakeOllama):
    """Healthy enough to start, dead by the first call -- the tunnel's failure."""

    def do_POST(self) -> None:
        self.send_error(502, "tunnel gone")


def test_a_run_that_lost_its_server_does_not_look_finished(tmp_path):
    """It happened: the tunnel died at call 28 of 110, the run exited 0, and the
    predictions file it left behind would have scored as a finished rung 1."""
    import run_detect
    server = HTTPServer(("127.0.0.1", 0), _DyingOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    code = run_detect.main([
        "--dataset", "halka", "--model", "test-model",
        "--base-url", f"http://127.0.0.1:{server.server_port}",
        "--timeout", "2", "--limit", "2", "--run-id", "dead",
        "--out", str(tmp_path), "--quiet",
    ])
    server.shutdown()
    assert code == 1
    manifest = json.loads((tmp_path / "dead" / "config.json").read_text())
    assert manifest["complete"] is False
    assert manifest["call_failures"] == 2


def test_a_clean_run_is_marked_complete(tmp_path):
    import run_detect
    assert run_detect.main([
        "--dataset", "halka", "--stub", "silent", "--limit", "2",
        "--run-id", "fine", "--out", str(tmp_path), "--quiet",
    ]) == 0
    assert json.loads((tmp_path / "fine" / "config.json").read_text())["complete"] is True
