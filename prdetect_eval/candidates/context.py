"""Repository view an enumerator is allowed to use.

Scoring must be reproducible from ``eval.jsonl`` alone, but enumeration is not
scoring: a real pull-request bot can read the whole checkout, and one of the
duplicate-code labels has its twin in a file the pull request never touches. So
the tree resolves changed files from the case overlay and everything else from
the pristine baseline, which is exactly the post-merge content.

Only the *diff* limits what may be reported -- that filter lives in
``keep_in_diff`` and is applied by every enumerator, because Bitbucket can only
annotate lines the pull request changed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import pyunits
from schema import Candidate, Case, Span

BASELINE = Path(__file__).resolve().parents[2] / "repo" / "repo"


@lru_cache(maxsize=None)
def _baseline_file(relative: str) -> str | None:
    path = BASELINE / relative
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _baseline_paths() -> tuple[str, ...]:
    if not BASELINE.is_dir():
        return ()
    return tuple(sorted(
        path.relative_to(BASELINE).as_posix() for path in BASELINE.rglob("*.py")
    ))


class Tree:
    """Post-merge content of the repository for one case."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self._units: dict[str, list[pyunits.Unit]] = {}

    def read(self, relative: str) -> str | None:
        if relative in self.case.head_files:
            return self.case.head_files[relative]
        if relative in self.case.deleted_files:
            return None
        return _baseline_file(relative)

    def paths(self) -> list[str]:
        names = set(_baseline_paths()) | set(self.case.head_files)
        return sorted(names - set(self.case.deleted_files))

    def units(self, relative: str) -> list[pyunits.Unit]:
        if relative not in self._units:
            source = self.read(relative)
            self._units[relative] = pyunits.units(source, relative) if source else []
        return self._units[relative]

    def lines(self, relative: str) -> list[str]:
        source = self.read(relative)
        return source.split("\n") if source else []

    @property
    def changed(self) -> list[str]:
        """Changed Python files that still exist, in a stable order."""
        return sorted(
            name for name in self.case.head_files
            if name.endswith(".py") and self.case.added_lines.get(name)
        )

    def touched(self, span: Span) -> bool:
        return self.case.touches(span)


def enclosing_region(tree: Tree, filename: str, line: int) -> Span:
    """Innermost function around ``line``, falling back to its class, then the line itself."""
    all_units = tree.units(filename)
    unit = pyunits.enclosing(all_units, line, "function") or pyunits.enclosing(all_units, line, "class")
    return unit.span if unit else Span(filename, line, line)


def keep_in_diff(tree: Tree, items: Iterable[Candidate]) -> Iterator[Candidate]:
    """Drop candidates the pull request does not touch.

    The corpus baseline ships eight identical ``summarize_changes`` bodies and
    several unreferenced helpers; without this filter every enumerator would
    drown in pre-existing findings that no reviewer can act on anyway.
    """
    for candidate in items:
        if tree.touched(candidate.focus) or tree.touched(candidate.region):
            yield candidate
