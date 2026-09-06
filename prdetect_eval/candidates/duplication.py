"""Arm B: structural hashing for duplicate code.

Copies are renamed, so a text hash finds nothing. The enumerator normalises each
function to its control-flow skeleton -- identifiers, literals and docstrings
erased, structure kept -- and groups equal skeletons across the whole tree,
because one corpus copy has its twin in a file the pull request never touches.

The baseline ships eight identical ``summarize_changes`` bodies, so without the
diff filter this enumerator would report the same pre-existing pair on nearly
every case. That is the filter earning its place, not an accident.
"""
from __future__ import annotations

import ast
from collections import defaultdict

import pyunits
from schema import Candidate, Case, Span

from .context import Tree, keep_in_diff

MIN_LINES = 6
MIN_NODES = 12


class _Skeleton(ast.NodeTransformer):
    """Erase every name and literal, keep every branch."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # Keep the type: swapping a string for a number is a real difference.
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.name = "_"
        node.returns = None
        node.decorator_list = []
        return node


def _skeleton(unit: pyunits.Unit) -> tuple[str, int] | None:
    node = unit.node
    if node is None or unit.kind != "function" or unit.length < MIN_LINES:
        return None
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # a docstring is not structure
    if not body:
        return None
    module = ast.Module(body=[ast.fix_missing_locations(_Skeleton().visit(ast.parse(ast.unparse(statement))))
                              for statement in body], type_ignores=[])
    size = sum(1 for _ in ast.walk(module))
    if size < MIN_NODES:
        return None
    return ast.dump(module, annotate_fields=False), size


def enumerate_case(case: Case, tree: Tree) -> list[Candidate]:
    groups: dict[str, list[pyunits.Unit]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for path in tree.paths():
        for unit in tree.units(path):
            skeleton = _skeleton(unit)
            if skeleton is None:
                continue
            digest, size = skeleton
            groups[digest].append(unit)
            sizes[digest] = size

    found: list[Candidate] = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        for unit in members:
            twins = [other for other in members if other is not unit]
            found.append(Candidate(
                case_id=case.case_id, type="duplicate_code", detector="duplication.ast_hash",
                focus=Span(unit.file, unit.def_line, unit.end_line), region=unit.span, symbol=unit.qualname,
                evidence={
                    "kind": "structural_clone", "nodes": sizes[digest], "copies": len(members),
                    "twins": [
                        {"file": twin.file, "line": twin.def_line, "symbol": twin.qualname}
                        for twin in twins[:4]
                    ],
                },
            ))
    return list(keep_in_diff(tree, found))
