"""Stage [2] of the pipeline: enumerate sites worth asking the model about.

Three arms, per the architecture: population counters where the defect is an
absence (``authz``, ``race_condition``), static patterns where the code is
present and has a shape (``sql_injection``, ``dead_code``, ``duplicate_code``),
and explicit hunk reading, which needs a model and so belongs to a later phase.

Nothing here claims a defect. Every enumerator emits the whole population --
scoped and unscoped reads alike -- and records the evidence that decides the
verdict. The judgment is always the model's.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from schema import Candidate, Case

from . import authz, concurrency, deadcode, duplication, injection
from .context import Tree

ENUMERATORS: dict[str, Callable[[Case, Tree], list[Candidate]]] = {
    "authz": authz.enumerate_case,
    "race_condition": concurrency.enumerate_case,
    "sql_injection": injection.enumerate_case,
    "dead_code": deadcode.enumerate_case,
    "duplicate_code": duplication.enumerate_case,
}

ARM = {
    "authz": "A population counter",
    "race_condition": "A population counter",
    "sql_injection": "B static pattern",
    "dead_code": "B static pattern",
    "duplicate_code": "B static pattern",
}


def enumerate_case(case: Case, types: Sequence[str] | None = None) -> list[Candidate]:
    tree = Tree(case)
    wanted = list(ENUMERATORS) if types is None else [name for name in types if name in ENUMERATORS]
    found: list[Candidate] = []
    for name in wanted:
        found.extend(ENUMERATORS[name](case, tree))
    return _dedupe(found)


def enumerate_all(cases: Iterable[Case], types: Sequence[str] | None = None) -> list[Candidate]:
    return [candidate for case in cases for candidate in enumerate_case(case, types)]


def _dedupe(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Two enumerators can flag the same line; the model should see it once."""
    seen: set[tuple] = set()
    unique = []
    for candidate in candidates:
        key = (candidate.type, candidate.focus, candidate.region, candidate.detector)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique
