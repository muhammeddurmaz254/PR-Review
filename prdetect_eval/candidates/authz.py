"""Arm A: population counter for authorization.

A missing organization filter cannot be grepped for -- absence has no pattern.
So the enumerator counts the population instead: every place in the diff where
tenant scoping *could* be applied becomes a candidate, whether or not it looks
wrong. Deciding is the model's job.

Four unit kinds make up the population:

1. touched view classes, which either declare ``permission_classes`` or do not
2. touched ORM reads, which either carry an organization predicate or do not
3. touched calls whose organization value comes from the request body or query
   string rather than the authenticated user
4. touched functions that accept a scoping parameter and never read it

Each candidate carries the sibling table -- what the neighbouring endpoints in
the same file do -- because that comparison is the only evidence that separates
a deliberately global endpoint from a broken one.
"""
from __future__ import annotations

import ast
from typing import Iterator

import pyunits
from schema import Candidate, Case, Span

from .context import Tree, enclosing_region, keep_in_diff

VIEW_BASES = ("ViewSet", "APIView", "GenericAPIView", "ListAPIView", "RetrieveAPIView")
SCOPE_NAMES = ("organization_id", "organization", "org_id", "user", "request")
REQUEST_SOURCES = ("query_params", "data", "GET", "POST")
ORM_TERMINALS = ("get", "filter", "all", "first", "last", "exclude", "none", "create", "count", "exists")


def _chain(node: ast.AST) -> list[str]:
    """Attribute names of a dotted expression, outermost last."""
    parts: list[str] = []
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Subscript):
            current = current.value
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            break
        else:
            break
    return list(reversed(parts))


def is_view_class(unit: pyunits.Unit) -> bool:
    return unit.kind == "class" and (
        any(base.endswith(VIEW_BASES) for base in unit.bases) or unit.name.endswith(("ViewSet", "View"))
    )


def _declares(unit: pyunits.Unit, attribute: str) -> bool:
    node = unit.node
    if node is None:
        return False
    return any(
        isinstance(statement, (ast.Assign, ast.AnnAssign))
        and any(target.id == attribute for target in _targets(statement))
        for statement in node.body
    )


def _targets(statement: ast.stmt) -> Iterator[ast.Name]:
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    for target in targets:
        if isinstance(target, ast.Name):
            yield target


def _orm_reads(node: ast.AST) -> Iterator[ast.Call]:
    """Calls that reach the database through a Django manager."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            chain = _chain(child.func)
            if "objects" in chain and chain[-1] in ORM_TERMINALS:
                yield child


def _request_derived(node: ast.AST) -> Iterator[ast.AST]:
    """Expressions reading straight from the request body or query string."""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in REQUEST_SOURCES:
            base = child.value
            if isinstance(base, ast.Name) and base.id in ("request", "self"):
                yield child


def narrows_to_organization(node: ast.AST) -> bool:
    """True when this code restricts a queryset to one organization.

    The obvious test -- does the word "organization" appear -- reads
    ``select_related("organization")`` as tenant scoping, which is a join hint and
    scopes nothing. It agrees with ``filter(organization_id=...)`` and with a
    selector that takes the organization as an argument, so the evidence handed
    to the model would be identical for a broken read, a filtered one and one
    scoped a layer down. That is the whole judgment, so it is worth an AST walk.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        for keyword in child.keywords:
            if keyword.arg and "organization" in keyword.arg:
                return True
        for argument in child.args:
            if isinstance(argument, ast.Attribute) and "organization" in argument.attr:
                return True
            if isinstance(argument, ast.Name) and "organization" in argument.id:
                return True
    return False


def _unused_scope_parameters(unit: pyunits.Unit) -> list[str]:
    node = unit.node
    if node is None or unit.kind != "function":
        return []
    parameters = [argument.arg for argument in node.args.args + node.args.kwonlyargs]
    scoped = [name for name in parameters if name in SCOPE_NAMES or name.endswith("organization_id")]
    if not scoped:
        return []
    used = {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
    used |= {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
    return [name for name in scoped if name not in used]


def sibling_table(tree: Tree, filename: str) -> list[dict]:
    """What every view class in this file declares, defect or not.

    ``authz`` is the one type where the neighbours are the evidence: an endpoint
    without ``permission_classes`` is only suspicious when its siblings have it.
    """
    rows = []
    for unit in tree.units(filename):
        if not is_view_class(unit):
            continue
        methods = [
            member for member in tree.units(filename)
            if member.kind == "function" and member.class_name == unit.name
        ]
        rows.append({
            "class": unit.name,
            "line": unit.def_line,
            "permission_classes": _declares(unit, "permission_classes"),
            "authentication_classes": _declares(unit, "authentication_classes"),
            "methods": [member.name for member in methods],
            "scoped_queryset": any(
                member.node is not None and narrows_to_organization(member.node)
                for member in methods if member.name == "get_queryset"
            ),
        })
    return rows


def enumerate_case(case: Case, tree: Tree) -> list[Candidate]:
    found: list[Candidate] = []
    for filename in tree.changed:
        all_units = tree.units(filename)
        siblings = sibling_table(tree, filename)

        for unit in all_units:
            if not is_view_class(unit):
                continue
            found.append(Candidate(
                case_id=case.case_id, type="authz", detector="authz.view_class",
                # The class header itself is often unchanged when a guard is
                # deleted, so the corpus anchors on the first body line; focus
                # there too or the two conventions disagree at k=3.
                focus=Span(filename, unit.body_line, unit.body_line),
                region=unit.span, symbol=unit.qualname,
                evidence={
                    "permission_classes": _declares(unit, "permission_classes"),
                    "siblings": siblings, "kind": "view_class",
                },
            ))

        for unit in all_units:
            if unit.kind != "function" or unit.node is None:
                continue
            region = unit.span
            for call in _orm_reads(unit.node):
                found.append(Candidate(
                    case_id=case.case_id, type="authz", detector="authz.orm_read",
                    focus=Span(filename, call.lineno, call.lineno), region=region,
                    symbol=unit.qualname,
                    evidence={
                        "scoped": narrows_to_organization(call), "kind": "orm_read",
                        "siblings": siblings,
                    },
                ))
            for node in _request_derived(unit.node):
                found.append(Candidate(
                    case_id=case.case_id, type="authz", detector="authz.request_scope",
                    focus=Span(filename, node.lineno, node.lineno), region=region,
                    symbol=unit.qualname,
                    evidence={"kind": "request_derived", "source": node.attr, "siblings": siblings},
                ))
            unused = _unused_scope_parameters(unit)
            if unused:
                found.append(Candidate(
                    case_id=case.case_id, type="authz", detector="authz.unused_scope_param",
                    focus=Span(filename, unit.body_line, unit.body_line), region=region,
                    symbol=unit.qualname,
                    evidence={"kind": "unused_scope_parameter", "parameters": unused},
                ))
    return list(keep_in_diff(tree, found))
