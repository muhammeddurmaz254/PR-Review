"""Arm A: population counter for race conditions.

A lost update has no syntax of its own -- the defect is a *missing* lock, so the
population is every read-modify-write the diff introduces. The enumerator finds
an ORM read bound to a name, then a later write to that same name in the same
function, and hands the pair to the model together with the facts that decide
the verdict: whether the row was locked, whether a transaction wraps the block,
and whether the write went through an atomic database expression instead.
"""
from __future__ import annotations

import ast
from typing import Iterator

import pyunits
from schema import Candidate, Case, Span

from .authz import _chain
from .context import Tree, keep_in_diff

READ_TERMINALS = ("get", "first", "last", "filter", "all", "select_for_update")
WRITE_METHODS = ("save", "update", "delete", "add", "remove", "set", "create")
ATOMIC_MARKERS = ("transaction.atomic", "atomic")


def _reads(node: ast.AST) -> Iterator[tuple[str, ast.Call]]:
    """``name = <manager chain>`` assignments, the read half of the pattern."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = child.value
        while isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            chain = _chain(value.func)
            if "objects" in chain and chain[-1] in READ_TERMINALS:
                yield target.id, value


def _writes(node: ast.AST, name: str) -> list[ast.Call]:
    """Later calls that persist ``name`` back to the database."""
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            base = child.func.value
            if isinstance(base, ast.Name) and base.id == name and child.func.attr in WRITE_METHODS:
                found.append(child)
    return found


def _mutations(node: ast.AST, name: str) -> list[ast.stmt]:
    """Assignments to ``name.<attribute>`` between the read and the write."""
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == name:
                    found.append(child)
        elif isinstance(child, ast.AugAssign):
            target = child.target
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == name:
                found.append(child)
    return found


def _uses_f_expression(node: ast.AST) -> bool:
    """``F("count") + 1`` pushes the arithmetic into the database and is safe."""
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "F"
        for child in ast.walk(node)
    )


def enumerate_case(case: Case, tree: Tree) -> list[Candidate]:
    found: list[Candidate] = []
    for filename in tree.changed:
        lines = tree.lines(filename)
        for unit in tree.units(filename):
            if unit.kind != "function" or unit.node is None:
                continue
            body = "\n".join(lines[unit.def_line - 1:unit.end_line])
            atomic = any(marker in decorator for decorator in unit.decorators for marker in ATOMIC_MARKERS) \
                or "transaction.atomic" in body
            for name, call in _reads(unit.node):
                writes = [write for write in _writes(unit.node, name) if write.lineno > call.lineno]
                if not writes:
                    continue
                read_text = "\n".join(lines[call.lineno - 1:(call.end_lineno or call.lineno)])
                found.append(Candidate(
                    case_id=case.case_id, type="race_condition", detector="concurrency.read_modify_write",
                    focus=Span(filename, call.lineno, call.lineno), region=unit.span, symbol=unit.qualname,
                    evidence={
                        "variable": name,
                        "locked": "select_for_update" in read_text,
                        "in_transaction": atomic,
                        "atomic_expression": _uses_f_expression(unit.node),
                        "write_lines": [write.lineno for write in writes],
                        "mutation_lines": [
                            statement.lineno for statement in _mutations(unit.node, name)
                            if statement.lineno > call.lineno
                        ],
                    },
                ))
    return list(keep_in_diff(tree, found))
