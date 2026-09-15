"""One finding per code site.

A reviewer reads a place in a file, not a list of claims: two claims a few lines
apart are one comment however differently they are named. `one_per_site` keeps
the most confident claim within `RADIUS` lines of each other in one file. Where a
change really does put two separate defects at one site, it publishes the one the
model was surer of.

`fill_gaps` adds the rules' findings only where nothing is published nearby, so a
rule never replaces the model's name for a site the model already covered.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

RADIUS = 5


def one_per_site(predictions: Iterable[dict], radius: int = RADIUS) -> list[dict]:
    """Keep the most confident claim at each site, in the order given.

    Works on prediction rows, so it needs only `case_id`, `file`, `line` and
    `confidence`.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)
    for claim in predictions:
        by_case[claim.get("case_id", "")].append(claim)
    keep: set[int] = set()
    for claims in by_case.values():
        chosen: list[dict] = []
        for claim in sorted(claims, key=_strength):
            if any(_same_site(other, claim, radius) for other in chosen):
                continue
            chosen.append(claim)
            keep.add(id(claim))
    return [claim for claims in by_case.values() for claim in claims if id(claim) in keep]


def _strength(claim: dict) -> tuple:
    """Most confident first; ties broken by line so the choice is stable."""
    return (-(claim.get("confidence") or 0.0), claim.get("line") or 0)


def _same_site(kept: dict, claim: dict, radius: int) -> bool:
    if kept.get("file") != claim.get("file"):
        return False
    a, b = kept.get("line"), claim.get("line")
    if a is None or b is None:
        return False
    return abs(a - b) <= radius


def fill_gaps(published: Sequence[dict], extra: Iterable[dict], radius: int = RADIUS) -> list[dict]:
    """`published`, plus each of `extra` that lands where nothing is published yet."""
    kept = list(published)
    for claim in extra:
        if any(_same_site(other, claim, radius) for other in kept):
            continue
        kept.append(claim)
    return kept
