"""Stage [5]: hold each finding to the line it says it is about.

Stage [6] was written first and measured a net loss: asking the model a second
time whether its own claim survives the evidence reproduces the mistake that
produced the claim. But the refutations it got *right* had one thing in common --
they were mechanical. The argument was on the next line; the character was in
the string. Nothing about them needed judgement, only comparison.

So this stage asks the detector to copy the line it is accusing, and compares
that copy against what the prompt actually printed. Three outcomes:

``anchored``   the quote is the line named. Nothing to do.
``snapped``    the quote is somewhere else in that file, at exactly one line.
               The claim is about real code and the number is wrong, so the
               number is corrected -- localization for free.
``unsupported`` the quote is in none of the lines the model was shown. It is
               describing code that was not there.

The comparison is on the code, not on the printing: the gutter, indentation and
one level of backslash escaping are normalised away, because all three were
observed to differ between what the prompt shows and what a model copies back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from schema import Case

from . import contract
from . import pack as pack_module

GUTTER = re.compile(r"^\s*\d*\s*[+-]?\s*\| ", re.M)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", GUTTER.sub("", text).replace("\\\\", "\\")).strip().lower()


@dataclass(frozen=True)
class Decision:
    """What the filter did with one report, and why."""

    report: contract.Report
    verdict: str
    line: int
    detail: str = ""

    @property
    def kept(self) -> bool:
        return self.verdict != "unsupported"


def resolve(reports: Sequence[contract.Report], case: Case, min_quote: int = 4) -> list[Decision]:
    """One decision per report, in order."""
    numbered, deleted = pack_module.shown_lines(case)
    decisions: list[Decision] = []
    for report in reports:
        quote = normalise(report.quote)
        if len(quote) < min_quote:
            # Too short to identify anything: a bare `)` matches half the file.
            decisions.append(Decision(report, "unchecked", report.line, "quote too short"))
            continue
        lines = numbered.get(report.file, {})
        if not lines:
            decisions.append(Decision(report, "unchecked", report.line, "file was not shown"))
            continue
        here = [normalise(text) for text in lines.get(report.line, ())]
        if any(one and (quote in one or one in quote) for one in here):
            decisions.append(Decision(report, "anchored", report.line))
            continue
        hits = [number for number, texts in sorted(lines.items())
                if any(quote in normalise(text) for text in texts)]
        if len(hits) == 1:
            decisions.append(Decision(report, "snapped", hits[0], f"quote is at line {hits[0]}"))
            continue
        if hits:
            decisions.append(Decision(report, "ambiguous", report.line,
                                      f"quote matches {len(hits)} lines"))
            continue
        if any(quote in normalise(text) for text in deleted.get(report.file, ())):
            # The defect is the deletion; the quote is a line with no number.
            decisions.append(Decision(report, "deleted-line", report.line, "quote is a removed line"))
            continue
        decisions.append(Decision(report, "unsupported", report.line,
                                  "quote appears nowhere in the lines shown"))
    return decisions


def apply(decisions: Sequence[Decision]) -> list[contract.Report]:
    """The surviving reports, with snapped line numbers corrected."""
    from dataclasses import replace
    return [replace(d.report, line=d.line) for d in decisions if d.kept]
