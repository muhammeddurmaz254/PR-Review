"""Reviewing what the first read skipped, and nothing else."""
from __future__ import annotations

from prdetect.cases import Case, is_code
from prdetect.detect import contract, continuation


def two_file_case() -> tuple[Case, list[str]]:
    head = {"app/a.py": "x = 1\n", "app/b.py": "y = 2\n", "README.md": "doc\n"}
    item = Case(case_id="c", pr_title="", pr_description="", changed_files=tuple(head), deleted_files=(),
                added_lines={name: frozenset({1}) for name in head}, head_files=head, context_files={}, diff="")
    return item, sorted(name for name in head if is_code(name))


def test_a_silent_first_read_is_not_reopened():
    item, _ = two_file_case()
    assert continuation.targets(item, []) == []


def test_only_the_unreported_code_files_are_asked_about():
    item, code = two_file_case()
    assert continuation.targets(item, [code[0]]) == code[1:]
    assert continuation.targets(item, code) == []


def test_the_trailer_records_what_is_done_and_names_only_what_is_left():
    _, code = two_file_case()
    text = continuation.trailer([{"file": code[0], "line": 3, "type": "x", "message": "m"}], code[1:])
    head, _, rest = text.partition("# STILL TO REVIEW")
    assert f"`{code[0]}`:3" in head
    assert f"`{code[0]}`" not in rest
    assert all(f"`{name}`" in rest for name in code[1:])
    assert "empty `findings` list" in text
    assert "probably" not in text and "likely" not in text


def test_a_report_outside_the_files_asked_about_is_dropped():
    reports = [contract.Report("a.py", 1, "x", "t", 0.9), contract.Report("b.py", 2, "x", "t", 0.9)]
    assert [r.file for r in continuation.within(reports, ["b.py"])] == ["b.py"]


def test_a_line_already_claimed_is_not_claimed_twice():
    reports = [contract.Report("a.py", 1, "x", "t", 0.9), contract.Report("a.py", 2, "y", "t", 0.8),
               contract.Report("a.py", 2, "z", "t", 0.7)]
    assert [(r.line, r.type) for r in continuation.fresh(reports, {("a.py", 1)})] == [(2, "y")]
