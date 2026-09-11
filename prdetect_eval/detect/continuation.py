"""Stage [4b]: after a finding, review the changed files the first read skipped.

The detector writes one finding per pull request, not one per defect. In
halka and demo_repo, 88% and 89% of defective pull requests get exactly one --
and every one of their pull requests carries exactly one defect, so the habit
costs nothing there and was never measured. zincir_dev carries two defects per
defective pull request and the model still writes one: the volume alone caps
its recall at 0.50, before a single judgement is wrong. Seven of its ten misses
sit in a pull request the model *did* speak about, in a changed file it never
reported on.

So this stage asks one narrower question, only where it can pay: the first read
found something, and some changed files carry no report -- review those. The
pack stays whole. That is what separates it from the split measured in §C9,
which cut a diff into hunk groups and would sever the two-file relationships
eight demo_repo defects live in.

What it deliberately does not do is tell the model that a pull request with one
defect probably has another. That is a prior, and moving a prior slides the
model along the speak-or-stay-silent curve §D14 measured seventeen prompts on;
none lifted it. The system prompt and its "an empty list is a normal answer"
stay exactly as they were. Only the attention moves.

Exposure was sized before it was written. halka: fifteen calls, two of them on
clean pull requests that already carry a false alarm, and none of its four
misses in reach -- all four sit in pull requests the model said nothing about.
zincir_dev: six calls, no clean pull request, seven misses in reach.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from schema import Case

from . import contract
from . import pack as pack_module


def targets(case: Case, reported: Iterable[str]) -> list[str]:
    """Changed code files the first read left without a report, or none.

    Nothing when the first read stayed silent: a silent pull request is the
    speak-or-stay-silent decision, which this stage does not reopen.
    """
    seen = set(reported)
    if not seen:
        return []
    return sorted(name for name in case.head_files if pack_module.is_code(name) and name not in seen)


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
    """Only reports in the files this call was asked about.

    A report on an already-reviewed file is either a repeat or a second opinion
    the first read did not give, and neither is what the call was for.
    """
    allowed = set(remaining)
    return [report for report in reports if report.file in allowed]
