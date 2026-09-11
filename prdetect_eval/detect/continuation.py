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


# --- the role question -------------------------------------------------------
#
# Stage [4b] measured that pointing the model at the unread files is not
# enough: in six of seven calls it looked at a test file or a module of
# defaults under the general question and saw nothing. So the question itself
# changes, and only for files whose role can be read off the file without a
# label. Two can. A test file, by the convention every Python test runner
# uses. A module of constants, where every top-level statement is an
# UPPER_CASE assignment -- no threshold, so nothing here was fitted to where
# the labels sit. "Configuration" in general could not be read off a file:
# zincir's `settings.py` is a function building a dict, structurally closer to
# its `deadletter.py` than to its `defaults.py`, and a rule that caught one
# without the other would have been drawn around the answers.
#
# MEASURED, AND IT LOST -- which is why `--mode role` is not the default.
# Against the best chain (deletions, [4b], snap): zincir_dev 8/3/8 became
# 8/6/8 and 8/5/8 on two runs of the same command; halka stayed 44/9/4 on
# twelve calls and twelve empty answers. Thirty-four zincir calls, six of the
# eight remaining misses in reach, and not one found. What the question did
# produce was the same accusation on both twins: `wrong_assertion_target` on
# `tests/test_transport.py` in `bus-01-temiz`, where it is false, in both runs,
# and in `bus-01-kusurlu` one line off the label in one run and not at all in
# the other. A question that raises the same flag on the defective file and
# its clean twin has not learned the defect; it has lowered the bar for the
# file. Pull-request balanced accuracy fell from 91.0% to 85.4%.
#
# Those two calls are also where two runs of one command first disagreed:
# identical prompts, identical order, temperature 0, seed 7, and two of
# thirty-four answers differ. The detector's own call has repeated exactly
# every time it was checked; this one does not, so a single role run is not a
# measurement of it.

from .roles import is_constants_module, is_test_file


def role_targets(case: Case) -> list[tuple[str, str]]:
    """Every changed code file whose role can be read off it, with that role."""
    out = []
    for name in sorted(case.head_files):
        if not pack_module.is_code(name):
            continue
        if is_test_file(name):
            out.append((name, "test"))
        elif is_constants_module(case.head_files[name]):
            out.append((name, "constants"))
    return out


QUESTIONS = {
    "test": ("It is a test file, and a test is judged by what it proves. A test is "
             "defective when it would still pass if the behaviour it names were broken, "
             "when it checks something other than what its name promises, or when it "
             "leaves state behind that another test will read. Ask that of each test "
             "this change touches, including an assertion or a reset the change removed."),
    "constants": ("It is a module of constants, and a constant is judged by the code that "
                  "reads it. A constant is defective when its value is unsafe for that "
                  "code, when it contradicts another value it has to agree with, or when "
                  "this change removed or renamed a name that code still reads. Ask that "
                  "of each value this change touches."),
}


def focus(path: str, role: str, reported: Sequence[dict]) -> str:
    """What follows the pack: what is recorded, then one file and its question."""
    done = [f"- `{c['file']}`:{c['line']} {c['type']} -- {c.get('message') or c.get('title', '')}"
            for c in reported]
    head = (["# ALREADY REPORTED", "",
             "A first review of this pull request reported the findings below. They "
             "are recorded; do not report them again.", "", *done, ""] if done else [])
    return "\n".join([
        *head,
        "# FOCUS",
        "",
        f"Review only `{path}`. {QUESTIONS[role]}",
        "",
        "Report a defect in this file the way the instructions above describe, or "
        "return an empty `findings` list.",
    ])


def fresh(reports: Sequence[contract.Report], claimed: Iterable[tuple[str, int]]) -> list[contract.Report]:
    """Reports on a line nothing has claimed yet -- one comment per line, as `dedupe` rules."""
    taken = set(claimed)
    out = []
    for report in reports:
        if (report.file, report.line) not in taken:
            taken.add((report.file, report.line))
            out.append(report)
    return out
