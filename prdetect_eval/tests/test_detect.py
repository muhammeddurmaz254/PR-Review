"""Tests for the detector path: packing, the response contract, the client.

These cover the failures that would look exactly like a bad model: a code block
whose line numbers do not match the file, a prompt that drifts between calls and
quietly disables prefix caching, or a parser that discards malformed answers
instead of counting them.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from statistics import median
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters import load_cases
from detect import client as client_module
from detect import anchor, challenge, consequence, contract, pack, prompt
from detect import scope as scope_module
from corpora import NARROW, REPOSITORIES, WIDE
from corpora import case as corpus_case
from corpora import cases as corpus_cases

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


@pytest.fixture(scope="session", params=list(REPOSITORIES))
def dataset(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def packs(dataset) -> list[pack.Pack]:
    cases = corpus_cases(dataset)
    return [pack.build(case, dataset) for case in cases]


def test_every_case_is_asked(dataset, packs):
    cases = corpus_cases(dataset)
    assert {p.case_id for p in packs} == {c.case_id for c in cases}


def test_prompt_prefix_is_byte_identical(packs):
    """Prefix caching is the load-bearing optimization; drift here is silent."""
    assert len({p.system for p in packs}) == 1


def test_each_prompt_describes_its_own_print_format(dataset):
    """Explaining the wrong rendering is worse than explaining none. Every
    repository is fetched as whole files, so no prompt describes renumbered hunks."""
    text = prompt.system(dataset)
    assert "Lines marked `+`" in text and "`-` and no number" not in text


def test_prompt_lists_exactly_the_dataset_types(dataset):
    text = prompt.system(dataset)
    assert re.findall(r"^- `([^`]+)` -- ", text, re.M) == prompt.types(dataset)


def test_code_block_line_numbers_match_the_file(dataset):
    """The model answers with file:line, so renumbering would make every hit wrong."""
    for case in corpus_cases(NARROW):
        item = pack.build(case, NARROW)
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
    marked = 0
    for case in corpus_cases(NARROW):
        item = pack.build(case, NARROW)
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


def test_the_diffs_added_lines_are_the_cases_added_lines(dataset):
    """The fetcher reads added lines off the Bitbucket diff; the pack's own hunk
    parser, read independently, must agree with it line for line. That parser was
    written for a diff format that named the file on a bare line, so on a git
    diff it keeps the `+++ b/` marker in the name."""
    for case in corpus_cases(dataset):
        marked: dict[str, set[int]] = {}
        for hunk in pack.parse_diff(case.diff):
            name = hunk.filename.removeprefix("+++ b/")
            for row in hunk.lines:
                if row[5:8] == " + ":
                    marked.setdefault(name, set()).add(int(row[:5]))
        expected = {name: set(lines) for name, lines in case.added_lines.items() if lines}
        assert marked == expected, case.case_id


def test_a_pack_is_empty_only_when_the_change_touches_no_code(dataset, packs):
    """All twenty-five clean SWRBench cases once had an empty diff and an empty
    file list, so their zero false alarms measured an empty prompt rather than a
    model. Since only source is reviewed a change to a manifest alone is empty
    too -- which is allowed, but it must not be scored, and no case carrying a
    scored label may be one."""
    cases = {case.case_id: case for case in corpus_cases(dataset)}
    for item in packs:
        case = cases[item.case_id]
        if item.shown_lines == 0:
            assert not case.reviewable, item.case_id
            assert not case.scored_labels, item.case_id
        else:
            assert case.reviewable, item.case_id


def test_clean_and_defective_packs_are_the_same_size(dataset, packs):
    """Length must not be a shortcut to the verdict."""
    cases = {case.case_id: case for case in corpus_cases(dataset)}
    defective = sorted(p.shown_lines for p in packs if cases[p.case_id].is_defective)
    clean = sorted(p.shown_lines for p in packs if not cases[p.case_id].is_defective)
    if not clean or not defective:
        pytest.skip("one-sided dataset")
    ratio = median(defective) / median(clean)
    assert 0.5 < ratio < 2.0, f"{dataset}: {median(defective)} vs {median(clean)} lines"


def test_every_label_is_printed_in_its_pack(dataset, packs):
    """A finding the pack never shows is unreachable, not hard."""
    by_id = {item.case_id: item for item in packs}
    for case in corpus_cases(dataset):
        text = by_id[case.case_id].user
        current = ""
        for label in case.scored_labels:
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
    for case in corpus_cases(NARROW):
        if not any(label.cross_file for label in case.labels):
            continue
        text = pack.build(case, NARROW).user
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
    assert response.thinking == ""
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


def test_the_verdict_is_decoded_after_the_reason():
    """Constrained decoding emits properties in schema order, and with thinking
    off that order is the only place the model can work. Measured with the
    verdict first: two of the three true positives it killed carried a reason
    that confirmed the claim the verdict had already rejected."""
    fields = list(challenge.VERDICT_SCHEMA["properties"])
    assert fields == ["reason", "quote", "verdict"]
    assert challenge.VERDICT_SCHEMA["required"] == fields
    assert "`verdict` comes last" in challenge.SYSTEM


def test_no_verdict_value_is_a_negation():
    """`refuted` was read as "I have something to say about this line": nine of
    twenty refutations carried a reason that agreed with the claim. Each value
    now names what the excerpt does, and only one of them removes a finding."""
    values = challenge.VERDICT_SCHEMA["properties"]["verdict"]["enum"]
    assert values == ["supports", "contradicts", "does_not_settle"]
    for value in ("supports", "does_not_settle", "anything-unrecognised", ""):
        assert challenge.parse(json.dumps({"verdict": value, "quote": "x", "reason": "r"}))[0] == "stands"
    assert challenge.parse(json.dumps(
        {"verdict": "contradicts", "quote": "x", "reason": "r"}))[0] == "refuted"


def test_the_excerpt_carries_the_detector_line_numbers(dataset):
    """The claim names a line; an excerpt renumbered against it proves nothing."""
    cases = corpus_cases(dataset)
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


def _case(dataset: str, source_case: str = ""):
    return corpus_case(dataset, source_case)


def test_shown_lines_are_exactly_what_the_pack_printed(dataset):
    """The filter compares against these; if they drift it rejects real findings."""
    for case in corpus_cases(dataset):
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


def test_a_line_that_comes_back_is_not_a_removal():
    """Half of zincir_dev's deleted lines are a settings table reordered.

    Rendering a moved line as a removal hands the model a "this is gone"
    candidate for something still there -- and on `ckpt-01-kusurlu` it would
    have handed it six, in the one case whose real defect is a key that is gone.
    """
    case = _case(WIDE, "authz-01-kusurlu")
    moved = [text for rows in pack.removed_lines(case).values() for _, text in rows]
    added = {line[1:].strip() for line in case.diff.split("\n")
             if line.startswith("+") and not line.startswith("+++")}
    assert moved, "this case is chosen because it deletes something"
    assert not [t for t in moved if t.strip() in added]


def test_the_whole_file_pack_prints_what_the_change_deleted():
    """The prompt has always asked for removals; only the hunk path answered it.

    The prompt says a removal "has no `+` line at all" and tells the model to
    quote the deleted line. On a corpus that ships whole files there
    was no deleted line in the pack to quote, so the instruction was
    unsatisfiable -- the same shape of gap as a response contract asking for a
    line number against a raw diff.
    """
    case = _case(WIDE, "authz-01-kusurlu")
    lean = pack.build(case, WIDE)
    full = pack.build(case, WIDE, with_deletions=True)
    assert not re.search(r"^\s+- \| ", lean.user, re.M), "off by default"
    assert re.search(r"^\s+- \| ", full.user, re.M)
    assert "Lines marked `-` were deleted" in full.user
    # Removals carry no line number, so they join no budget and no anchor.
    assert full.shown_lines == lean.shown_lines


def test_a_removal_is_quotable_only_when_the_pack_printed_it():
    """The gate must be told what the pack was told; otherwise it invents room.

    `deleted-line` accepts a quote that matches no numbered line. Granting it
    against a pack that printed no removals would admit a quote the model was
    never shown -- the gate would stop being a check on the pack.
    """
    case = _case(WIDE, "authz-01-kusurlu")
    _, deleted = pack.shown_lines(case, with_deletions=True)
    filename = next(iter(deleted))
    report = contract.Report(filename, 1, "x", "t", 0.9, quote=deleted[filename][0])
    assert not pack.shown_lines(case)[1], "silent unless asked"
    assert anchor.resolve([report], case)[0].verdict == "unsupported"
    assert anchor.resolve([report], case, with_deletions=True)[0].verdict == "deleted-line"


def test_the_challenger_is_shown_the_same_removals_as_the_detector():
    """A challenger blind to a deletion refutes the finding that read it.

    Measured on `log-01-kusurlu`: the detector reported the leak at the
    surviving `counters.reset()`, exactly the line the corpus labels, and the
    challenger -- shown head lines only -- answered that the fixture "explicitly
    resets the state" and refuted it. Same shape as the `conf-03` failure that
    put the counted facts into `challenge.build`.
    """
    case = _case(WIDE, "authz-01-kusurlu")
    filename, rows = next(iter(pack.removed_lines(case).items()))
    at = rows[0][0]
    lean = challenge.excerpt(case, filename, at, radius=12)
    full = challenge.excerpt(case, filename, at, radius=12, with_deletions=True)
    assert not [r for r in lean if r[:5].strip() == "" and r.lstrip().startswith("-")]
    assert [r for r in full if r[:5].strip() == "" and r.lstrip().startswith("-")]
    # The window is chosen by printed number; a removal must not push a
    # numbered line out of it.
    assert {r[:5].strip() for r in lean} <= {r[:5].strip() for r in full}


def test_a_removal_is_anchored_where_it_was_removed_from():
    """A deleted line has no number, so the model cannot name the right one.

    On `ckpt-01-kusurlu` it named the line above the gap and the corpus anchors
    the line below it: the same defect, scored at tolerance 0 as a miss and a
    false alarm at once. The pack knows where the removal sat, so the gate
    corrects it, as `snapped` already does for a quote found at another number.
    """
    case = _case(WIDE, "authz-01-kusurlu")
    filename, rows = next(iter(pack.removed_lines(case).items()))
    at, text = rows[0]
    report = contract.Report(filename, at + 4, "x", "t", 0.9, quote=text)
    decision = anchor.resolve([report], case, with_deletions=True)[0]
    assert decision.verdict == "deleted-line" and decision.kept
    assert decision.line == at, "the gate knows the position; the model cannot"
    assert anchor.apply([decision])[0].line == at


def test_an_absent_quote_is_never_used_to_drop_a_finding(dataset):
    """Versions before v4 do not ask for one; the filter must stay out of the way."""
    case = _case(dataset)
    decisions = anchor.resolve([contract.Report("a.py", 1, "x", "t", 0.9)], case)
    assert decisions[0].verdict == "unchecked" and decisions[0].kept


def test_the_prompt_asks_for_the_quote_and_the_schema_requires_it():
    assert "`quote` is the code on that line" in prompt.system(WIDE)
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
    for dataset in REPOSITORIES:
        cases = corpus_cases(dataset)
        densest = max(len(case.scored_labels) for case in cases)
        assert densest <= 3, f"{dataset} carries {densest} required labels on one case"


def test_splitting_is_off_by_default_in_build(dataset):
    """`build` is the whole pull request; only `split` with a limit divides it."""
    for case in corpus_cases(dataset):
        assert pack.build(case, dataset).parts == 1


def test_a_split_keeps_every_line_exactly_once(dataset):
    """A hunk dropped or duplicated between excerpts is a silent recall loss."""
    for case in corpus_cases(dataset):
        whole = pack.build(case, dataset)
        parts = pack.split(case, dataset, max_lines=120)
        assert sum(p.shown_lines for p in parts) == whole.shown_lines, case.case_id
        assert {p.part for p in parts} == set(range(1, len(parts) + 1))


def test_whole_file_cases_are_never_split(dataset):
    """Cross-file defects exist only between two changed files; separating them
    would make those unreachable however small the limit."""
    for case in corpus_cases(dataset):
        assert len(pack.split(case, dataset, max_lines=1)) == 1


def test_every_label_is_reachable_after_a_split(dataset):
    """The split must not put a finding in no excerpt at all."""
    for case in corpus_cases(dataset):
        text = "\n".join(p.user for p in pack.split(case, dataset, max_lines=120))
        for label in case.scored_labels:
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


def test_halka_carries_the_repository_behind_the_diff():
    """Thirty-three of forty-nine defects are only legible against code the pull
    request does not touch; a pack without it cannot reach them."""
    cases = corpus_cases(WIDE)
    assert all(case.context_files for case in cases)
    for case in cases:
        assert not (set(case.context_files) & set(case.head_files)), case.case_id


def test_with_repo_adds_the_unchanged_files_and_nothing_else():
    case = corpus_cases(WIDE)[0]
    lean, full = pack.build(case, WIDE), pack.build(case, WIDE, context=("*",))
    assert lean.shown_lines == full.shown_lines, "changed-code accounting must not move"
    assert full.estimated_tokens > lean.estimated_tokens * 3
    assert lean.user in full.user or "# WHAT THIS PULL REQUEST CHANGED" in full.user
    for filename in case.context_files:
        if pack.is_code(filename):
            assert f"# FILE {filename}   (unchanged)" in full.user
        assert filename not in lean.user


def test_the_anchor_filter_sees_the_repository_when_the_pack_does():
    """Otherwise every quote taken from an unchanged file is called invented."""
    case = corpus_cases(WIDE)[0]
    filename, line, text = next(
        (name, n, row)
        for name in sorted(case.context_files)
        for n, row in enumerate(case.context_files[name].split("\n"), 1)
        if len(row.strip()) > 24
    )
    report = contract.Report(filename, line, "sql_injection", "t", 0.9, quote=text)
    assert anchor.resolve([report], case)[0].verdict == "unchecked"
    assert anchor.resolve([report], case, context=("*",))[0].verdict == "anchored"


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
            "--dataset", WIDE, "--stub", "silent", "--resume",
            "--run-id", "half", "--out", str(tmp_path), "--quiet",
        ])
    assert "different system prompt" in str(raised.value)


def test_resume_reuses_the_answers_already_on_disk(tmp_path):
    import run_detect
    args = ["--dataset", WIDE, "--stub", "silent", "--limit", "4",
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
        "--dataset", WIDE, "--model", "test-model",
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
        "--dataset", WIDE, "--stub", "silent", "--limit", "2",
        "--run-id", "fine", "--out", str(tmp_path), "--quiet",
    ]) == 0
    assert json.loads((tmp_path / "fine" / "config.json").read_text())["complete"] is True


def test_only_labels_the_pack_can_show_are_in_scope():
    """The corpus marks eight types outside `birincil_kapsam`, but the reason is
    the product's rule-id vocabulary rather than the label, so every one of them
    is scored. The single exclusion is a different thing: a label in a manifest,
    which the pack no longer prints and so cannot be found."""
    cases = corpus_cases(WIDE)
    labels = [label for case in cases for label in case.labels]
    assert len(labels) == 49
    for label in labels:
        assert label.in_scope == pack.is_code(label.span.file), label.finding_id
    assert sum(not label.in_scope for label in labels) == 1


def test_context_can_be_narrowed_to_a_pattern():
    """Carrying everything was measured and lost, so the mechanism has to be able
    to carry a little: the conventions doc is 560 tokens against the repo's
    twelve thousand."""
    case = corpus_cases(WIDE)[0]
    narrow = pack.build(case, WIDE, context=("halka/common/*.py",))
    everything = pack.build(case, WIDE, context=("*",))
    assert "# FILE halka/common/http.py   (unchanged)" in narrow.user
    assert narrow.estimated_tokens < everything.estimated_tokens / 2
    for filename in case.context_files:
        if not filename.startswith("halka/common/"):
            assert f"# FILE {filename}   (unchanged)" not in narrow.user


def test_a_narrowed_context_narrows_the_anchor_filter_too():
    case = corpus_cases(WIDE)[0]
    outside, line, text = next(
        (name, n, row)
        for name in sorted(case.context_files) if not name.startswith("docs/")
        for n, row in enumerate(case.context_files[name].split("\n"), 1)
        if len(row.strip()) > 24
    )
    report = contract.Report(outside, line, "sql_injection", "t", 0.9, quote=text)
    assert anchor.resolve([report], case, context=("docs/*.md",))[0].verdict == "unchecked"
    assert anchor.resolve([report], case, context=("*",))[0].verdict == "anchored"


def test_one_comment_per_line():
    """A bot that leaves two comments on one line is worse to read than one --
    an argument available before seeing any data, which is what separates this
    from a rule fitted to the answers. Measured: fires once in 188 cases, costs
    no true positive on any of the three corpora."""
    hedged = [contract.Report("a.py", 84, "swallowed_exception", "t", 0.9, quote="pass"),
              contract.Report("a.py", 84, "crossfile_error_propagation", "u", 0.8, quote="pass")]
    kept = contract.dedupe(hedged)
    assert len(kept) == 1 and kept[0].type == "swallowed_exception"
    # A different line is a different claim, and a different file is too.
    spread = hedged + [contract.Report("a.py", 85, "broad_except", "t", 0.9),
                       contract.Report("b.py", 84, "broad_except", "t", 0.9)]
    assert len(contract.dedupe(spread)) == 3


def test_the_legacy_field_order_is_exactly_what_was_measured():
    """Six prompt versions were measured under it; their numbers mean nothing
    under another order."""
    schema = contract.response_schema(["authz"], quote=True)["properties"]["findings"]["items"]
    assert list(schema["properties"]) == ["file", "line", "type", "title", "confidence", "quote"]
    assert schema["required"] == list(schema["properties"])
    plain = contract.response_schema(["authz"])["properties"]["findings"]["items"]
    assert list(plain["properties"]) == ["file", "line", "type", "title", "confidence"]


# --- stage [6]: a one-file excerpt cannot settle a two-file claim


def test_a_cross_file_claim_cannot_be_refuted_from_one_file():
    assert not challenge.settleable("crossfile_unit_mismatch")
    assert not challenge.settleable("crossfile_ownership")


def test_every_other_kind_of_claim_can_be():
    for kind in ("sql_injection", "missing_lock", "F.2 Logic", "business_logic"):
        assert challenge.settleable(kind)


# --- [4a] / [4b]: the detector reports, a second call names ------------------


def test_an_open_answer_has_no_type_field():
    schema = contract.response_schema(
        [], quote=True, order=contract.OPEN_ORDER)["properties"]["findings"]["items"]
    assert list(schema["properties"]) == ["file", "line", "title", "confidence", "quote"]
    assert "type" not in schema["properties"]


def test_the_catalogue_holds_every_configured_kind():
    from detect import naming
    catalogue = naming.pool()
    for dataset in REPOSITORIES:
        for name in prompt.types(dataset):
            assert name in catalogue, name
    assert len(catalogue) == 74


def test_naming_may_refuse_every_candidate():
    """A name that does not fit is a wrong answer that reads like a right one, and
    an unreadable reply must not invent one either."""
    from detect import naming
    schema = naming.schema(["sql_injection", "xss"])
    assert naming.NONE in schema["properties"]["type"]["enum"]
    assert list(schema["properties"]) == ["reason", "type"]
    assert naming.parse(json.dumps({"reason": "r", "type": naming.NONE})) == ("", "r")
    assert naming.parse(json.dumps({"reason": "r", "type": "xss"})) == ("xss", "r")
    for broken in ("", "not json", "[]"):
        assert naming.parse(broken)[0] == ""


def test_naming_shows_only_the_shortlist():
    from detect import naming
    catalogue = naming.pool()
    candidates = ["sql_injection", "xss"]
    text = naming.build({"title": "t", "quote": "q", "file": "a.py", "line": 3},
                        candidates, catalogue)
    assert "`sql_injection`" in text and "`xss`" in text
    assert f"`{naming.NONE}`" in text
    for name in catalogue:
        if name not in candidates:
            assert f"`{name}`" not in text, name


def test_the_namer_refuses_to_search_blind():
    """A prediction row calls the finding's words `message`; reading `title` off
    it silently gave every finding the same shortlist and scored 12% where
    retrieval alone allows 91%."""
    import run_name
    from detect import naming
    catalogue = naming.pool()
    blind = naming.rank({"title": "", "quote": ""}, catalogue)
    seeing = naming.rank({"title": "Redirect target fetched without allow-list validation",
                          "quote": "requests.get(url)"}, catalogue)
    assert blind != seeing, "an empty query must not look like a real one"
    assert blind == list(catalogue)[:len(blind)], "the empty case is the catalogue head"


def test_the_client_keeps_the_thinking_it_is_given(fake_server):
    """Thinking was off from the first model run on the belief that a schema
    cannot constrain it. Ollama returns it in its own field and the answer stays
    clean, so the belief was wrong -- and the reasoning is worth storing: it is
    the artefact to read when a finding looks unexplainable."""
    class _Thinker(_FakeOllama):
        def do_POST(self) -> None:
            self._send({"message": {"content": contract.render([]), "thinking": "step one"},
                        "prompt_eval_count": 5, "done": True})

    server = HTTPServer(("127.0.0.1", 0), _Thinker)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = client_module.OllamaClient(
        model="test-model", base_url=f"http://127.0.0.1:{server.server_port}")
    response = client.complete("s", "u", contract.response_schema(["authz"]))
    server.shutdown()
    assert response.thinking == "step one"
    assert contract.parse(response.text) == ([], [])


def test_repo_facts_are_silent_unless_they_discriminate():
    """Two of the three kinds first written were deleted before costing a run:
    counting same-prefix siblings fired on 33 defective cases and 38 clean, and
    reporting any unreferenced symbol on 32 and 41. What is left speaks on two
    cases in a hundred and ten, both of them defective."""
    from detect import facts
    cases = corpus_cases(WIDE)
    speaking = [case for case in cases if facts.collect(case)]
    assert len(speaking) == 2
    assert all(case.is_defective for case in speaking)


def test_repo_facts_only_ask_about_names_the_change_defines():
    """A fact about code the pull request never touches is an invitation to hunt."""
    from detect import facts
    for case in corpus_cases(WIDE):
        collected = facts.collect(case)
        if not collected:
            continue
        touched = "\n".join(
            "\n".join(case.source_lines(name)[line - 1] for line in sorted(lines)
                      if line <= len(case.source_lines(name)))
            for name, lines in case.added_lines.items() if name in case.head_files)
        for fact in collected:
            assert fact.subject in touched, fact.subject


def test_the_facts_block_states_answers_not_code():
    from detect import facts
    case = _case(WIDE, "conf-03-kusurlu")
    lean = pack.build(case, WIDE)
    with_facts = pack.build(case, WIDE, with_facts=True)
    assert "WHAT THE REPOSITORY SAYS" in with_facts.user
    assert "WHAT THE REPOSITORY SAYS" not in lean.user
    assert "measurements, not accusations" in with_facts.user
    # A fact is a line, not a file: the block costs tens of tokens, not thousands.
    assert with_facts.estimated_tokens - lean.estimated_tokens < 200


def test_the_challenger_is_told_what_was_counted():
    """It refuted a true positive it could not have settled: the claim was that a
    timeout duplicates a value in another module, and the excerpt was one file."""
    from detect import facts
    case = _case(WIDE, "conf-03-kusurlu")
    counted = [fact.render() for fact in facts.collect(case)]
    assert counted, "the fact that settles this claim has to exist"
    claim = {"file": "halka/integrations/client.py", "line": 12,
             "title": "Timeout value duplicated from settings source", "type": "duplicated_config"}
    without = challenge.build(claim, ["   12   | ZAMAN_ASIMI_SANIYE = 5.0"])
    with_facts = challenge.build(claim, ["   12   | ZAMAN_ASIMI_SANIYE = 5.0"], counted)
    assert "OUTBOUND_TIMEOUT_SECONDS" not in without
    assert "OUTBOUND_TIMEOUT_SECONDS" in with_facts
    assert "not read from the excerpt above" in with_facts


# ---------------------------------------------------------------- stage [5] scope


def test_a_pull_request_with_no_code_gets_no_findings(dataset):
    """The two claims written from a title and description alone were both false.

    A pack that carries no code cannot ground a report, and the challenge stage
    cannot tell -- an empty excerpt neither shows nor rules out anything, so it
    refuted the claim on `conf-01-kusurlu` and let the identical claim on
    `conf-01-temiz` stand. The rule belongs one stage earlier and mechanically.
    """
    cases = [c for c in corpus_cases(dataset) if not c.reviewable]
    if not cases:
        pytest.skip(f"{dataset} has no case without reviewable code")
    for case in cases:
        report = contract.Report(file=next(iter(case.head_files), "x.py"), line=1,
                                 type="secrets", title="anything at all", confidence=1.0)
        decisions = scope_module.resolve([report], case)
        assert not decisions[0].kept, case.case_id
        assert "no code" in decisions[0].detail


def test_the_scope_gate_never_drops_a_label(dataset):
    """The gate is only free if no labelled defect lies outside its own diff."""
    checked = 0
    for case in corpus_cases(dataset):
        for label in case.labels:
            if not label.in_scope:
                continue
            report = contract.Report(file=label.span.file, line=label.span.start_line,
                                     type=label.type, title="", confidence=1.0)
            checked += 1
            assert scope_module.resolve([report], case)[0].kept, (case.case_id, label.type)
    assert checked


def test_the_scope_gate_reads_the_file_not_the_line():
    """A pure deletion has no added line to point at and is still this PR's defect."""
    case = _case(NARROW)
    changed = next(iter(case.changed_files))
    report = contract.Report(file=changed, line=10 ** 6, type="secrets", title="",
                             confidence=1.0)
    assert scope_module.resolve([report], case)[0].kept


# ------------------------------------------------------- stage [6b] consequence


def test_only_an_absent_harm_removes_a_finding():
    """Every other answer, including one this module does not know, keeps it."""
    for value in ("style_only", "none"):
        assert consequence.harmless(value)
    for value in ("wrong_behaviour", "unsafe_access", "resource_cost",
                  "", "unparseable", "severe", "MAYBE"):
        assert not consequence.harmless(value)


def test_an_unreadable_answer_keeps_the_detectors_report():
    """A server hiccup must not be scored as a precision gain."""
    for text in ("", "not json", "[]", '{"harm": null}'):
        harm, trigger, effect = consequence.parse(text)
        assert not consequence.harmless(harm)


def test_the_schema_puts_the_run_before_the_verdict():
    """Generation order is schema order; the label must name work already done."""
    assert list(consequence.HARM_SCHEMA["properties"]) == ["trigger", "consequence", "harm"]
    assert consequence.HARM_SCHEMA["properties"]["harm"]["enum"][-2:] == ["style_only", "none"]


def test_a_claim_of_harm_must_point_at_a_run():
    """`concrete` is to a survival what `honours` is to a refutation."""
    assert consequence.concrete("a member of another org calls void",
                                "the invoice is voided")
    for trigger, effect in (("none", "the invoice is voided"),
                            ("", "the invoice is voided"),
                            ("a caller", "none"),
                            ("N/A", "n/a"),
                            ("yok", "hicbiri")):
        assert not consequence.concrete(trigger, effect), (trigger, effect)


def test_the_two_verify_nodes_are_asked_different_questions():
    """The split is the point: same evidence, different question.

    Stage [6a] agreed with twelve of the thirteen false alarms it was shown, and
    its own reasons say why -- they were accurate. A second node that merely
    reworded "contradict" would inherit that ceiling.
    """
    assert "contradict" in challenge.SYSTEM
    assert "contradict" not in consequence.SYSTEM
    assert "goes wrong" in consequence.SYSTEM
    claim = {"file": "a.py", "line": 3, "type": "wrong_argument", "title": "t"}
    rows = ["   3 + | f(x)"]
    assert consequence.build(claim, rows) != challenge.build(claim, rows)
    # Same evidence, so a difference in the answers is a difference in the question.
    for payload in (consequence.build(claim, rows), challenge.build(claim, rows)):
        assert "f(x)" in payload and "wrong_argument" in payload


# --- the catalogue: one list for every repository ---------------------------


def test_the_catalogue_carries_no_synonym_of_a_coarse_corpus_name():
    """A catalogue with synonyms was measured at -0.25 F1 (D10). demo_repo used
    to label with these coarse names; its labels now use catalogue names, and the
    coarse ones must not come back into the catalogue beside them."""
    coarse = {"authz", "business_logic", "injection", "race_condition", "data_exposure",
              "secrets", "error_handling", "idempotency"}
    assert not set(prompt.types(WIDE)) & coarse


def test_every_catalogue_name_has_a_definition():
    definitions = prompt.definitions(WIDE)
    assert list(definitions) == prompt.types(WIDE)
    assert all(text.strip() for text in definitions.values())


# --- per-file review --------------------------------------------------------

def test_per_file_makes_one_pack_for_each_changed_code_file():
    case = _case(WIDE)
    packs = pack.split(case, WIDE, prompt.PROMPT_VERSION, per_file=True)
    files = sorted(n for n in case.head_files if pack.is_code(n))
    assert len(packs) == len(files)
    assert [p.part for p in packs] == list(range(1, len(files) + 1))
    assert all(p.parts == len(files) for p in packs)
    for p, name in zip(packs, files):
        assert f"# FILE {name}" in p.user
        assert sum(f"# FILE {other}" in p.user for other in files) == 1


def test_a_per_file_pack_names_the_other_changed_files():
    case = _case(WIDE)
    files = sorted(n for n in case.head_files if pack.is_code(n))
    if len(files) < 2:
        return
    packs = pack.split(case, WIDE, prompt.PROMPT_VERSION, per_file=True)
    assert f"`{files[1]}`" in packs[0].user
    assert "being reviewed separately" in packs[0].user


def test_whole_request_packing_is_untouched():
    one = pack.split(_case(WIDE), WIDE, prompt.PROMPT_VERSION)
    assert len(one) == 1


