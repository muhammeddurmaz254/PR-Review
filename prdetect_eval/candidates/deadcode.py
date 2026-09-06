"""Arm B: static enumeration for dead code.

Two shapes, both mechanical: statements that follow an unconditional exit in the
same block, and module-private definitions nothing in the repository names. The
second needs a whole-repository reference count, which is why the enumerator
reads the tree rather than only the diff -- a helper added by this pull request
may well be called from a file the pull request never touched.

The enumerator does not decide. A handler reached through ``getattr`` and a
Celery ``shared_task`` both look unreferenced here, and the corpus contains
exactly those traps.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from functools import lru_cache

import pyunits
from schema import Candidate, Case, Span

from .context import Tree, keep_in_diff

TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Names reached indirectly: a decorator can register a function that nothing calls.
INDIRECT_DECORATORS = ("shared_task", "task", "receiver", "register", "route", "app.task", "celery")


@lru_cache(maxsize=512)
def _identifiers(source: str) -> Counter:
    return Counter(IDENTIFIER.findall(source))


def _reference_counts(tree: Tree) -> Counter:
    """Identifier frequency across the whole post-merge tree."""
    total: Counter = Counter()
    for path in tree.paths():
        source = tree.read(path)
        if source:
            total.update(_identifiers(source))
    return total


def _unreachable_blocks(node: ast.AST):
    """First statement after an unconditional exit, per block, with the block end."""
    for parent in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list) or len(block) < 2:
                continue
            for index, statement in enumerate(block[:-1]):
                if isinstance(statement, TERMINAL):
                    tail = block[index + 1:]
                    yield tail[0], tail[-1]
                    break


def enumerate_case(case: Case, tree: Tree) -> list[Candidate]:
    found: list[Candidate] = []
    counts = _reference_counts(tree)
    for filename in tree.changed:
        source = tree.read(filename)
        if not source:
            continue
        module = pyunits.parse(source)
        if module is None:
            continue
        all_units = tree.units(filename)

        for first, last in _unreachable_blocks(module):
            end = last.end_lineno or last.lineno
            enclosing = pyunits.enclosing(all_units, first.lineno, "function")
            found.append(Candidate(
                case_id=case.case_id, type="dead_code", detector="deadcode.after_terminal",
                focus=Span(filename, first.lineno, end),
                region=enclosing.span if enclosing else Span(filename, first.lineno, end),
                symbol=enclosing.qualname if enclosing else "",
                evidence={"kind": "statements_after_exit", "lines": [first.lineno, end]},
            ))

        for unit in all_units:
            if unit.class_name and unit.kind == "function":
                continue  # methods are reached through their class, not by name
            # One occurrence is the definition itself; anything more is a reference.
            if counts.get(unit.name, 0) > 1:
                continue
            found.append(Candidate(
                case_id=case.case_id, type="dead_code", detector="deadcode.unreferenced",
                focus=Span(filename, unit.def_line, unit.def_line), region=unit.span, symbol=unit.qualname,
                evidence={
                    "kind": "no_reference_in_tree", "decorators": list(unit.decorators),
                    "indirect_decorator": any(
                        marker in decorator for decorator in unit.decorators for marker in INDIRECT_DECORATORS
                    ),
                    "exported": not unit.name.startswith("_"),
                },
            ))
    return list(keep_in_diff(tree, found))
