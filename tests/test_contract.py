"""The detector's answer: its schema, and how it is read back."""
import json

import pytest

from prdetect.detect import contract


def test_the_schema_limits_the_kind_and_requires_every_field_in_order():
    schema = contract.response_schema(["missing_authz_check", "xss"])
    items = schema["properties"]["findings"]["items"]
    assert items["properties"]["type"]["enum"] == ["missing_authz_check", "xss"]
    assert list(items["properties"]) == ["file", "line", "type", "title", "confidence", "quote"]
    assert items["required"] == list(items["properties"])


def test_parse_reads_a_valid_answer():
    reports, rejects = contract.parse(json.dumps({"findings": [
        {"file": "a/b.py", "line": 12, "type": "xss", "title": "t", "confidence": 0.7, "quote": "x = 1"}
    ]}))
    assert not rejects
    assert reports == [contract.Report("a/b.py", 12, "xss", "t", 0.7, "x = 1")]


def test_parse_accepts_an_empty_answer():
    assert contract.parse('{"findings": []}') == ([], [])


@pytest.mark.parametrize("payload", ["", "not json", "[]", '{"other": 1}', '{"findings": {}}'])
def test_parse_rejects_malformed_answers(payload):
    reports, rejects = contract.parse(payload)
    assert not reports and len(rejects) == 1


@pytest.mark.parametrize("finding", [
    {"file": "a.py", "line": 0}, {"file": "a.py", "line": "x"}, {"file": "", "line": 3}, {"line": 3},
])
def test_parse_counts_unusable_findings(finding):
    reports, rejects = contract.parse(json.dumps({"findings": [finding]}))
    assert not reports and len(rejects) == 1


def test_confidence_is_clamped():
    reports, _ = contract.parse(json.dumps({"findings": [
        {"file": "a.py", "line": 1, "type": "xss", "title": "", "confidence": 9.0}]}))
    assert reports[0].confidence == 1.0


def test_the_cap_keeps_the_most_confident():
    reports = [contract.Report("a.py", n, "x", "t", c) for n, c in ((1, 0.2), (2, 0.9), (3, 0.5), (4, 0.7))]
    assert [r.line for r in contract.cap(reports, 2)] == [2, 4]
    assert contract.cap(reports, 0) == reports
    assert len(contract.cap(reports, 10)) == 4


def test_one_report_per_line():
    hedged = [contract.Report("a.py", 84, "swallowed_exception", "t", 0.9, quote="pass"),
              contract.Report("a.py", 84, "crossfile_error_propagation", "u", 0.8, quote="pass")]
    kept = contract.dedupe(hedged)
    assert len(kept) == 1 and kept[0].type == "swallowed_exception"
    spread = hedged + [contract.Report("a.py", 85, "broad_except", "t", 0.9),
                       contract.Report("b.py", 84, "broad_except", "t", 0.9)]
    assert len(contract.dedupe(spread)) == 3


def test_render_round_trips():
    reports = [contract.Report("a.py", 3, "xss", "t", 0.5, "q")]
    assert contract.parse(contract.render(reports)) == (reports, [])
