"""Trivial heuristics every real number is reported against.

The corpus has smaller diffs than production pull requests, so any measured
score is optimistic. These strategies use no understanding of the code at all;
a detector that does not clear them has demonstrated nothing.

``hot_spot_memoriser`` and ``every_added_line`` deliberately cheat -- the first
reads the answer key to find each type's favourite location, the second reports
everything. They are ceilings, not competitors: if a detector cannot beat the
memoriser, the corpus concentrates its defects too narrowly to measure anything.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import pyunits
from schema import Case, Prediction, Span, in_scope_types


def _modal_type(cases: Sequence[Case]) -> str:
    counts = Counter(label.type for case in cases for label in case.scored_labels)
    return counts.most_common(1)[0][0] if counts else "authz"


def _first_touched(case: Case, filename: str) -> int | None:
    lines = case.added_lines.get(filename)
    return min(lines) if lines else None


def _changed_with_lines(case: Case) -> list[tuple[str, int]]:
    """Changed files that still exist at head, with their touched line count."""
    return [
        (filename, len(case.added_lines.get(filename, ())))
        for filename in sorted(case.head_files)
        if case.added_lines.get(filename)
    ]


def _point(case: Case, filename: str, line: int, defect_type: str, name: str) -> Prediction:
    return Prediction(
        case_id=case.case_id, span=Span(filename, line, line), type=defect_type,
        confidence=1.0, detector=name, stage="baseline",
        message=f"baseline {name}",
    )


def flag_everything(cases: Sequence[Case]) -> list[Prediction]:
    """Call every pull request defective and point at its first changed line."""
    defect_type = _modal_type(cases)
    predictions = []
    for case in cases:
        files = _changed_with_lines(case)
        if files:
            filename = files[0][0]
            predictions.append(_point(case, filename, _first_touched(case, filename), defect_type, "flag_everything"))
    return predictions


def spray_all_types(cases: Sequence[Case]) -> list[Prediction]:
    """First changed line, once per in-scope type: maximum type coverage, maximum noise."""
    predictions = []
    for case in cases:
        files = _changed_with_lines(case)
        if not files:
            continue
        filename = files[0][0]
        line = _first_touched(case, filename)
        for defect_type in sorted(in_scope_types(cases)):
            predictions.append(_point(case, filename, line, defect_type, "spray_all_types"))
    return predictions


def smallest_diff_file(cases: Sequence[Case]) -> list[Prediction]:
    """Always accuse the least-changed file. This scored 82 percent on the discarded corpus."""
    defect_type = _modal_type(cases)
    predictions = []
    for case in cases:
        files = _changed_with_lines(case)
        if files:
            filename = min(files, key=lambda item: (item[1], item[0]))[0]
            predictions.append(_point(case, filename, _first_touched(case, filename), defect_type, "smallest_diff_file"))
    return predictions


def largest_diff_file(cases: Sequence[Case]) -> list[Prediction]:
    """The complement of the above; the current corpus is meant to be a coin flip between them."""
    defect_type = _modal_type(cases)
    predictions = []
    for case in cases:
        files = _changed_with_lines(case)
        if files:
            filename = max(files, key=lambda item: (item[1], item[0]))[0]
            predictions.append(_point(case, filename, _first_touched(case, filename), defect_type, "largest_diff_file"))
    return predictions


def longest_changed_function(cases: Sequence[Case]) -> list[Prediction]:
    """Accuse the body of the longest function the pull request touches."""
    defect_type = _modal_type(cases)
    predictions = []
    for case in cases:
        touched = list(pyunits.touched_units(case))
        if not touched:
            continue
        filename, unit = max(touched, key=lambda item: (item[1].length, item[0], item[1].def_line))
        predictions.append(_point(case, filename, unit.body_line, defect_type, "longest_changed_function"))
    return predictions


def first_changed_line(cases: Sequence[Case]) -> list[Prediction]:
    """The single earliest touched line in the whole pull request."""
    defect_type = _modal_type(cases)
    predictions = []
    for case in cases:
        candidates = [
            (filename, min(lines)) for filename, lines in sorted(case.added_lines.items())
            if lines and filename in case.head_files
        ]
        if candidates:
            filename, line = candidates[0]
            predictions.append(_point(case, filename, line, defect_type, "first_changed_line"))
    return predictions


def hot_spot_memoriser(cases: Sequence[Case]) -> list[Prediction]:
    """Read the answer key: report each type at the location it most often occupies.

    This is leakage by construction. It exists to expose per-type file
    concentration in the corpus -- three of the four race_condition labels share
    one file, and five authz labels span two. A detector that merely matches
    this number has learned the corpus, not the defect.
    """
    hot: dict[str, Counter] = {}
    for case in cases:
        for label in case.scored_labels:
            hot.setdefault(label.type, Counter())[(label.span.file, label.span.start_line)] += 1
    spots = {defect_type: counter.most_common(1)[0][0] for defect_type, counter in hot.items()}
    predictions = []
    for case in cases:
        for defect_type, (filename, line) in sorted(spots.items()):
            predictions.append(_point(case, filename, line, defect_type, "hot_spot_memoriser"))
    return predictions


def every_added_line(cases: Sequence[Case]) -> list[Prediction]:
    """Report every touched line under every in-scope type: the recall ceiling of pure spray."""
    predictions = []
    for case in cases:
        for filename, lines in sorted(case.added_lines.items()):
            if filename not in case.head_files:
                continue
            for line in sorted(lines):
                for defect_type in sorted(in_scope_types(cases)):
                    predictions.append(_point(case, filename, line, defect_type, "every_added_line"))
    return predictions


BASELINES: dict[str, Callable[[Sequence[Case]], list[Prediction]]] = {
    "flag_everything": flag_everything,
    "spray_all_types": spray_all_types,
    "smallest_diff_file": smallest_diff_file,
    "largest_diff_file": largest_diff_file,
    "longest_changed_function": longest_changed_function,
    "first_changed_line": first_changed_line,
    "hot_spot_memoriser": hot_spot_memoriser,
    "every_added_line": every_added_line,
}

LEAKY = frozenset({"hot_spot_memoriser", "every_added_line"})
