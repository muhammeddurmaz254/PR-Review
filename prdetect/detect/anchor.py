"""Hold each claim to the line it quotes.

The detector copies the line it accuses, and the copy is compared with what its
pack printed:

``anchored``      the quote is the line named.
``snapped``       the quote is at exactly one other line of that file; the number is corrected.
``ambiguous``     the quote is at several lines; the claim is kept as it is.
``deleted-line``  the quote is a line the change deleted; the claim is placed where it was.
``unchecked``     the quote is too short to identify a line, or the file was not printed.
``unsupported``   the quote is in nothing that was printed; the claim is dropped.

The comparison ignores the gutter, indentation and one level of backslash
escaping, since a model copying a line back changes all three.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from prdetect.cases import Case
from prdetect.detect import contract
from prdetect.detect import pack as pack_module

GUTTER = re.compile(r"^\s*\d*\s*[+-]?\s*\| ", re.M)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", GUTTER.sub("", text).replace("\\\\", "\\")).strip().lower()


@dataclass(frozen=True)
class Decision:
    """What the check did with one report, and why."""

    report: contract.Report
    verdict: str
    line: int
    detail: str = ""

    @property
    def kept(self) -> bool:
        return self.verdict != "unsupported"


def resolve(reports: Sequence[contract.Report], case: Case, min_quote: int = 4,
            with_deletions: bool = False) -> list[Decision]:
    """One decision per report, in order.

    `with_deletions` must match the pack the answers came from: a quote of a
    deleted line is accepted only if the pack printed deleted lines.
    """
    numbered, deleted = pack_module.shown_lines(case, with_deletions)
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
        here = normalise(lines.get(report.line, ""))
        if here and (quote in here or here in quote):
            decisions.append(Decision(report, "anchored", report.line))
            continue
        hits = [number for number, text in sorted(lines.items()) if quote in normalise(text)]
        if len(hits) == 1:
            decisions.append(Decision(report, "snapped", hits[0], f"quote is at line {hits[0]}"))
            continue
        if hits:
            decisions.append(Decision(report, "ambiguous", report.line,
                                      f"quote matches {len(hits)} lines"))
            continue
        if any(quote in normalise(text) for text in deleted.get(report.file, ())):
            # The defect is the deletion, and a deleted line has no number the
            # model could name; the pack knows where it sat.
            where = [at for at, text in pack_module.removed_lines(case).get(report.file, ())
                     if quote in normalise(text)]
            at = where[0] if where else report.line
            decisions.append(Decision(report, "deleted-line", at,
                                      "quote is a removed line"
                                      + (f"; anchored at line {at}" if where else "")))
            continue
        decisions.append(Decision(report, "unsupported", report.line,
                                  "quote appears nowhere in the lines shown"))
    return decisions


def apply(decisions: Sequence[Decision]) -> list[contract.Report]:
    """The surviving reports, with corrected line numbers."""
    return [replace(d.report, line=d.line) for d in decisions if d.kept]
