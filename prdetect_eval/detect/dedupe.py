"""One finding per code site.

A reviewer reads a place in the file, not a list of claims, and two reports
that land on the same lines are one comment to them however differently they
are named. The detector does not know that: a shell call built from a caller's
string draws both `path_traversal` and `command_injection`, four lines apart,
and a read-modify-write draws `missing_lock` beside whatever else that
function is doing wrong.

MEASURED (D22). review/v13-universal + --mechanism, family rung, radius five:

    halka      41/11/7 -> 41/8/7   F1 0.820 -> 0.845
    zincir      7/2/9  ->  7/2/9   unchanged
    demo_repo  13/3/4  -> 13/2/4   F1 0.788 -> 0.812

    pooled F1 0.772 -> 0.792, precision 0.792 -> 0.836

Not one true finding is lost on any corpus, which is what lets this be on by
default. The radius does nothing between five and forty -- duplicate reports
sit within a couple of lines of each other and the next real finding is far
away -- so five is the whole range, not a tuned point. It is also safe on the
other prediction sets on disk: the per-corpus-catalogue control goes 16/3/1 ->
16/2/1 on demo_repo and is untouched on halka and zincir.

Restricting it to claims of the same family was measured and is WORSE (pooled
0.782): two of the duplicates it has to remove are named across families, and
the reviewer reading those lines still sees one comment.

The risk it carries, stated because the corpora happen not to trigger it:
halka's `perf-01-kusurlu` carries two required labels on the same line and
zincir's `bus-01-kusurlu` two a line apart. Where a change really does put two
separate defects at one site, this publishes the one the model was surer of.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

RADIUS = 5


def one_per_site(predictions: Iterable[dict], radius: int = RADIUS) -> list[dict]:
    """Keep the most confident claim at each site, in the given order.

    Works on the raw claim dicts the verify stage publishes, so it needs only
    `case_id`, `file`, `line` and `confidence`. Claims are returned in their
    original order: the caller's file is a record of the run, not a ranking.
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
    return [claim for claim in _flatten(by_case) if id(claim) in keep]


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


def _flatten(by_case: dict[str, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for claims in by_case.values():
        out.extend(claims)
    return out


def dropped(predictions: Sequence[dict], kept: Sequence[dict]) -> int:
    return len(predictions) - len(kept)

def fill_gaps(published: Sequence[dict], extra: Iterable[dict], radius: int = RADIUS) -> list[dict]:
    """Add `extra` findings only where nothing is published yet.

    The rules are a fallback for what the model does not see, not a competitor
    for what it does. Published beside it they win any site they share, because
    a rule's confidence is fixed and the model's is not -- and D28 measured
    what that costs: on `crypto-03` the model had named the weak digest at the
    line where `secret.literal` also fires, `one_per_site` kept the rule's name,
    and a true finding became a false alarm. Here the model keeps its site and
    a rule speaks only into silence.
    """
    kept = list(published)
    for claim in extra:
        if any(_same_site(other, claim, radius) for other in kept):
            continue
        kept.append(claim)
    return kept
