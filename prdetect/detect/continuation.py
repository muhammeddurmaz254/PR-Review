"""After a first read, review the changed files it left without a claim.

The detector tends to write one claim per pull request rather than one per
defect, so a pull request with defects in two files often gets one of them. Where
the first read said something and some changed code files carry no claim, the
same pack is sent again with a trailer naming what is already recorded and which
files are left. A pull request the first read stayed silent on is not reopened,
and the system prompt is unchanged.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from prdetect.cases import Case, is_code
from prdetect.detect import contract


def targets(case: Case, reported: Iterable[str]) -> list[str]:
    """Changed code files the first read left without a claim; none when it stayed silent."""
    seen = set(reported)
    if not seen:
        return []
    return sorted(name for name in case.head_files if is_code(name) and name not in seen)


def trailer(reported: Sequence[dict], remaining: Sequence[str]) -> str:
    """What follows the pack: what is already recorded, and what is left."""
    done = [f"- `{c['file']}`:{c['line']} {c['type']} -- {c.get('message') or c.get('title', '')}"
            for c in reported]
    return "\n".join([
        "# ALREADY REPORTED",
        "",
        "A first review of this pull request reported the findings below. They "
        "are recorded; do not report them again.",
        "",
        *done,
        "",
        "# STILL TO REVIEW",
        "",
        "These changed files have not been reviewed yet:",
        "",
        *(f"- `{name}`" for name in remaining),
        "",
        "Review only these files. Report a defect in them the way the "
        "instructions above describe, or return an empty `findings` list.",
    ])


def within(reports: Sequence[contract.Report], remaining: Sequence[str]) -> list[contract.Report]:
    """Only reports in the files this call was asked about."""
    allowed = set(remaining)
    return [report for report in reports if report.file in allowed]


def fresh(reports: Sequence[contract.Report], claimed: Iterable[tuple[str, int]]) -> list[contract.Report]:
    """Reports on a line nothing has claimed yet."""
    taken = set(claimed)
    out = []
    for report in reports:
        if (report.file, report.line) not in taken:
            taken.add((report.file, report.line))
            out.append(report)
    return out
