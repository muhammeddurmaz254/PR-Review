"""The two checks a claim passes before any model reads it again: its quote and its file."""
from __future__ import annotations

import pytest

from prdetect.detect import anchor, contract, pack, scope
from tests.repositories import NARROW, REPOSITORIES, WIDE, case, cases

DELETES = (WIDE, "PR-28")


@pytest.fixture(params=REPOSITORIES)
def repo(request) -> str:
    return request.param


def _long_line(item):
    numbered, _ = pack.shown_lines(item)
    filename = sorted(numbered)[0]
    return filename, numbered[filename]


def test_a_quote_that_matches_its_line_is_anchored(repo):
    item = case(repo)
    filename, lines = _long_line(item)
    line, text = next((n, t) for n, t in sorted(lines.items()) if len(t.strip()) > 12)
    report = contract.Report(filename, line, "x", "t", 0.9, quote=text)
    assert anchor.resolve([report], item)[0].verdict == "anchored"


def test_a_quote_from_another_line_snaps_to_it(repo):
    item = case(repo)
    filename, lines = _long_line(item)
    rows = [(n, t) for n, t in sorted(lines.items()) if len(t.strip()) > 12]
    unique = [(n, t) for n, t in rows
              if sum(1 for _, other in rows if anchor.normalise(t) in anchor.normalise(other)) == 1]
    line, text = unique[0]
    decision = anchor.resolve([contract.Report(filename, line + 500, "x", "t", 0.9, quote=text)], item)[0]
    assert decision.verdict == "snapped" and decision.line == line
    assert anchor.apply([decision])[0].line == line


def test_a_quote_that_was_never_shown_is_dropped(repo):
    item = case(repo)
    filename, _ = _long_line(item)
    report = contract.Report(filename, 1, "x", "t", 0.9, quote="raise NotImplementedError('nothing prints this')")
    decision = anchor.resolve([report], item)[0]
    assert decision.verdict == "unsupported" and not decision.kept
    assert anchor.apply([decision]) == []


def test_a_missing_quote_never_drops_a_claim(repo):
    decision = anchor.resolve([contract.Report("a.py", 1, "x", "t", 0.9)], case(repo))[0]
    assert decision.verdict == "unchecked" and decision.kept


def test_a_deleted_line_is_quotable_only_when_the_pack_printed_it():
    item = case(*DELETES)
    _, deleted = pack.shown_lines(item, with_deletions=True)
    filename = next(iter(deleted))
    report = contract.Report(filename, 1, "x", "t", 0.9, quote=deleted[filename][0])
    assert anchor.resolve([report], item)[0].verdict == "unsupported"
    assert anchor.resolve([report], item, with_deletions=True)[0].verdict == "deleted-line"


def test_a_deleted_line_is_anchored_where_it_was_removed():
    item = case(*DELETES)
    filename, rows = next(iter(pack.removed_lines(item).items()))
    at, text = rows[0]
    decision = anchor.resolve([contract.Report(filename, at + 4, "x", "t", 0.9, quote=text)], item,
                              with_deletions=True)[0]
    assert decision.verdict == "deleted-line" and decision.kept and decision.line == at


def test_a_pull_request_with_no_code_gets_no_claims(repo):
    unreviewable = [item for item in cases(repo) if not item.reviewable]
    assert unreviewable
    for item in unreviewable:
        report = contract.Report(next(iter(item.head_files), "x.py"), 1, "hardcoded_credential", "t", 1.0)
        decision = scope.resolve([report], item)[0]
        assert not decision.kept and "no code" in decision.detail


def test_the_scope_check_never_drops_a_label(repo):
    for item in cases(repo):
        for label in item.labels:
            if label.in_scope:
                report = contract.Report(label.span.file, label.span.start_line, label.type, "", 1.0)
                assert scope.resolve([report], item)[0].kept, label.finding_id


def test_the_scope_check_reads_the_file_not_the_line():
    item = case(NARROW)
    changed = next(name for name in item.changed_files if name.endswith(".py"))
    assert scope.resolve([contract.Report(changed, 10 ** 6, "x", "", 1.0)], item)[0].kept
    assert not scope.resolve([contract.Report("elsewhere.py", 1, "x", "", 1.0)], item)[0].kept
