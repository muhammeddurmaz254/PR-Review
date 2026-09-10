"""Stage [5]: refuse a report the pull request cannot be about.

Two rules, and neither of them reads the code. They exist because a measured
false alarm can be wrong before anyone looks at the line it names.

*A pull request with nothing to review gets no review.* Only source is packed,
so a change that touches nothing but a manifest or an example environment file
reaches the model as a title, a description and no code. Two of the thirteen
false alarms the broad taxonomy left on ``halka`` were written from prose alone:
``conf-01-temiz`` says in its description that dependencies were split into
``requirements.txt`` and ``requirements-dev.txt``, and the model reported
"dev-only packages left in runtime requirements" at line 1 of a file it had not
seen; ``temiz-11`` says an ``.env.example`` was added, and the model reported a
duplicated API key in ``src/config.py``, a file that is not in the pull request
at all. Both packs carried zero lines of code. The challenge stage cannot help
here -- it refuted the identical claim on ``conf-01-kusurlu`` and let this one
stand, because an empty excerpt settles nothing either way.

*A report must name a file this pull request changed.* A review comments on a
diff. A finding somewhere else may even be true, and it is still not this pull
request's, so it cannot be this pull request's defect. The weaker of the two
obvious spellings is deliberate: the file must be changed, not the line added.
``demo_repo`` labels a defect whose whole content is a deletion, and a
line-scoped rule would drop it for having no added line to point at.

Measured over the three corpora: no label of any of them lies outside its own
``changed_files``, so the second rule costs nothing that is known to be there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from schema import Case

from . import contract


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
