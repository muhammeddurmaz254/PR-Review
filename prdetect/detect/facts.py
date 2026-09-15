"""What the repository says about the names a change defines: counted, not shown.

Some defects turn on a fact about the whole repository rather than on anything in
the change -- whether a module-private name the change adds is used anywhere,
whether a constant it introduces already exists under another name. A reviewer
answers those by searching. The detector is shown only the changed files, so each
such question is answered here with `ast` over the repository at the pull
request, and stated in one line.

Nothing here shows code, and nothing judges: a count is not an accusation. Only
names the change itself defines are asked about, and a fact that would be true of
most changes is not stated at all, so the block is usually empty.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Sequence

from prdetect.cases import Case


@dataclass(frozen=True)
class Fact:
    """One settled question about the repository."""

    subject: str
    statement: str

    def render(self) -> str:
        return f"- `{self.subject}`: {self.statement}"


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _defined_here(case: Case) -> list[tuple[str, ast.AST]]:
    """Names the changed files define on lines the change added, with the defining node."""
    found: list[tuple[str, ast.AST]] = []
    for filename, source in sorted(case.head_files.items()):
        tree = _parse(source)
        if tree is None:
            continue
        added = case.added_lines.get(filename, frozenset())
        for node in ast.walk(tree):
            line = getattr(node, "lineno", None)
            if line is None or line not in added:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.append((node.name, node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.append((target.id, node))
    return found


def _sources(case: Case) -> dict[str, str]:
    return {**case.context_files, **case.head_files}


def _references(case: Case, name: str) -> int:
    """How many times this name is used as a name anywhere in the repository.

    Uses, not definitions. Attribute access counts; a dynamic call or a name
    reached through getattr is invisible, which is why the statement says how it
    was counted.
    """
    total = 0
    for filename, source in _sources(case).items():
        tree = _parse(source)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                total += 1
            elif isinstance(node, ast.Attribute) and node.attr == name:
                total += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                total += sum(1 for alias in node.names
                             if (alias.asname or alias.name.split(".")[-1]) == name)
    return total


def _literal(node: ast.AST) -> object | None:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        value = node.value.value
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return value
    return None


def _same_value_elsewhere(case: Case, name: str, value: object) -> list[str]:
    """Other module-level names assigned the same literal, anywhere in the repository."""
    others: list[str] = []
    for filename, source in sorted(_sources(case).items()):
        tree = _parse(source)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if node.value.value != value or type(node.value.value) is not type(value):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id != name:
                    others.append(f"{target.id} in {filename}")
    return others


def collect(case: Case, limit: int = 12) -> list[Fact]:
    """Every settled question worth stating about this change's own names."""
    if not case.head_files:
        return []
    facts: list[Fact] = []
    seen: set[str] = set()
    for name, node in _defined_here(case):
        if name in seen or name.startswith("__"):
            continue
        seen.add(name)

        # Only a module-private name. A public one nothing references may simply
        # be new, or reached through a router, a registry or a decorator no name
        # lookup can see. The leading underscore is the language's own statement
        # that nothing outside should call it, which is what makes the count mean
        # something.
        if name.startswith("_") and not name.startswith("__"):
            uses = _references(case, name)
            if uses <= (1 if isinstance(node, ast.Assign) else 0):
                facts.append(Fact(name, "declared module-private and used nowhere else that a "
                                        "name lookup can see (a dynamic call would not show)"))

        value = _literal(node)
        if value is not None:
            elsewhere = _same_value_elsewhere(case, name, value)
            if elsewhere:
                facts.append(Fact(name, f"the same value {value!r} is also assigned to "
                                        f"{', '.join(elsewhere[:3])}"))

    return facts[:limit]


def render(facts: Sequence[Fact]) -> list[str]:
    if not facts:
        return []
    return [
        "# WHAT THE REPOSITORY SAYS",
        "",
        "Counted over the whole repository at this revision -- not read, counted. "
        "Each line answers one question about a name this pull request defines. "
        "They are measurements, not accusations: whether any of them matters is "
        "yours to decide.",
        "",
        *(fact.render() for fact in facts),
        "",
    ]
