"""Layer 1: a diagnostic counts only when the pull request made its kind more frequent."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect.static import Diagnostic, introduced


def _d(rule: str, line: int, message: str = "m", file: str = "a.py") -> Diagnostic:
    return Diagnostic("ruff", rule, file, line, message)


def test_a_diagnostic_the_base_already_had_is_not_introduced_even_if_it_moved():
    assert introduced([_d("F401", 3)], [_d("F401", 9)]) == []


def test_a_kind_that_grew_reports_only_the_growth_preferring_added_lines():
    base = [_d("F841", 3)]
    head = [_d("F841", 5), _d("F841", 20)]
    new = introduced(base, head, {"a.py": frozenset({20})})
    assert [d.line for d in new] == [20]


def test_the_message_is_part_of_the_kind_so_a_new_name_is_new():
    assert introduced([_d("F821", 3, "`x` is undefined")], [_d("F821", 3, "`y` is undefined")]) != []


def test_only_the_first_line_of_a_message_and_its_spacing_define_the_kind():
    base = [_d("attr", 3, "Cannot access `id`\n  detail one")]
    head = [_d("attr", 7, "Cannot  access `id`\n  detail two")]
    assert introduced(base, head) == []
