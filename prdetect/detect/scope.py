"""Refuse a claim the pull request cannot be about.

Two rules, neither of which reads the code. A pull request that changes no source
reaches the model as a title and a description only, and a claim written from
those has nothing under it. And a review comments on its own change: a claim
about a file the pull request did not change may even be true, but it is not this
pull request's defect. The rule is about the file, not the line, because a defect
can be a deletion that leaves no added line to point at.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from prdetect.cases import Case
from prdetect.detect import contract


@dataclass(frozen=True)
class Decision:
    report: contract.Report
    kept: bool
    detail: str


def resolve(reports: Sequence[contract.Report], case: Case) -> list[Decision]:
    """One decision per report, in order, with the reason recorded either way."""
    if not case.reviewable:
        return [Decision(report, False, "the pull request shows no code to review")
                for report in reports]
    changed = set(case.changed_files)
    out: list[Decision] = []
    for report in reports:
        if report.file in changed:
            out.append(Decision(report, True, ""))
        else:
            out.append(Decision(report, False,
                                f"{report.file} is not a file this pull request changes"))
    return out


def apply(decisions: Sequence[Decision]) -> list[contract.Report]:
    return [decision.report for decision in decisions if decision.kept]
