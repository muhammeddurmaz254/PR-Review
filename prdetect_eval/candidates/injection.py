"""Arm B: static enumeration for SQL injection.

Here the code is present and has a shape, so a pattern search works. The subtle
part is that the defect is often not on the interpolation line: two of the three
corpus cases put the fault in the statement that *decides* what gets
interpolated -- an allow-list lookup whose fallback returns the raw input. So
the enumerator flags the interpolation and then walks one hop back to where each
interpolated name was last assigned inside the same function.
"""
from __future__ import annotations

import ast
from typing import Iterator

from schema import Candidate, Case, Span

from .authz import _chain
from .context import Tree, keep_in_diff

SQL_KEYWORDS = ("select ", "insert ", "update ", "delete ", "from ", "where ", "order by", "join ")
EXECUTORS = ("execute", "executemany", "raw", "RawSQL", "extra")


def _looks_like_sql(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SQL_KEYWORDS)


def _literal_text(node: ast.JoinedStr) -> str:
    return "".join(part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str))


def _interpolations(node: ast.AST) -> Iterator[tuple[ast.JoinedStr, ast.FormattedValue]]:
    """f-string holes inside a string that reads like SQL."""
    for child in ast.walk(node):
        if isinstance(child, ast.JoinedStr) and _looks_like_sql(_literal_text(child)):
            for part in child.values:
                if isinstance(part, ast.FormattedValue):
                    yield child, part


def _concatenations(node: ast.AST) -> Iterator[ast.AST]:
    """``"SELECT ... " + value`` and ``"SELECT ... %s" % value``."""
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Add, ast.Mod)):
            for side in (child.left, child.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, str) and _looks_like_sql(side.value):
                    yield child
                    break
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "format":
            base = child.func.value
            if isinstance(base, ast.Constant) and isinstance(base.value, str) and _looks_like_sql(base.value):
                yield child


def _executors(node: ast.AST) -> Iterator[ast.Call]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            chain = _chain(child.func)
            if chain and chain[-1] in EXECUTORS:
                yield child


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def _assignments(unit_node: ast.AST, names: set[str]) -> Iterator[tuple[str, ast.stmt]]:
    """Where each interpolated name was bound inside this function."""
    for child in ast.walk(unit_node):
        targets: list[ast.expr] = []
        if isinstance(child, ast.Assign):
            targets = list(child.targets)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            targets = [child.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                yield target.id, child


def enumerate_case(case: Case, tree: Tree) -> list[Candidate]:
    found: list[Candidate] = []
    for filename in tree.changed:
        for unit in tree.units(filename):
            if unit.kind != "function" or unit.node is None:
                continue
            region, interpolated = unit.span, set()

            for _, hole in _interpolations(unit.node):
                interpolated |= _names(hole)
                found.append(Candidate(
                    case_id=case.case_id, type="sql_injection", detector="injection.interpolation",
                    focus=Span(filename, hole.lineno, hole.lineno), region=region, symbol=unit.qualname,
                    evidence={"kind": "f_string_hole", "names": sorted(_names(hole))},
                ))

            for node in _concatenations(unit.node):
                interpolated |= _names(node)
                found.append(Candidate(
                    case_id=case.case_id, type="sql_injection", detector="injection.concatenation",
                    focus=Span(filename, node.lineno, node.lineno), region=region, symbol=unit.qualname,
                    evidence={"kind": "string_build", "names": sorted(_names(node))},
                ))

            for call in _executors(unit.node):
                parameterised = bool(call.args[1:]) or bool(call.keywords)
                found.append(Candidate(
                    case_id=case.case_id, type="sql_injection", detector="injection.executor",
                    focus=Span(filename, call.lineno, call.lineno), region=region, symbol=unit.qualname,
                    evidence={"kind": "executor", "parameterised": parameterised},
                ))

            # One hop back: the allow-list lookup that chooses the interpolated
            # value is the defect in two of the three corpus cases.
            for name, statement in _assignments(unit.node, interpolated):
                found.append(Candidate(
                    case_id=case.case_id, type="sql_injection", detector="injection.value_origin",
                    focus=Span(filename, statement.lineno, statement.lineno), region=region, symbol=unit.qualname,
                    evidence={"kind": "interpolated_value_origin", "name": name},
                ))
    return list(keep_in_diff(tree, found))
