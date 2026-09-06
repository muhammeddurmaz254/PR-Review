"""Line-preserving structural view of a Python file.

Both the dumb baselines and the candidate enumerators need the same question
answered: which function, class or statement covers this line? The standard
``ast`` module is enough for that and keeps the harness free of the generator
dependency, so the two never drift into agreeing by construction.

Every line number is one-based and every range includes both endpoints.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from schema import Span


@dataclass(frozen=True)
class Unit:
    """A function, method or class, addressed the way the corpus anchors are."""

    kind: str
    name: str
    qualname: str
    file: str
    def_line: int
    body_line: int
    end_line: int
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    class_name: str = ""
    node: ast.AST | None = field(default=None, repr=False, compare=False)

    @property
    def span(self) -> Span:
        return Span(self.file, self.def_line, self.end_line)

    @property
    def length(self) -> int:
        return self.end_line - self.def_line + 1

    def covers(self, line: int) -> bool:
        return self.def_line <= line <= self.end_line

    def source(self, lines: list[str]) -> str:
        return "\n".join(lines[self.def_line - 1:self.end_line])


def parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _name(node: ast.AST) -> str:
    """Dotted source name of a decorator or base expression, best effort."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return ""


def units(source: str, filename: str) -> list[Unit]:
    """Every top-level and nested class, function and method in definition order."""
    tree = parse(source)
    if tree is None:
        return []
    found: list[Unit] = []

    def walk(node: ast.AST, prefix: str, class_name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                is_class = isinstance(child, ast.ClassDef)
                body_line = child.body[0].lineno if child.body else child.lineno
                found.append(Unit(
                    kind="class" if is_class else "function",
                    name=child.name, qualname=qualname, file=filename,
                    def_line=child.lineno, body_line=body_line, end_line=child.end_lineno or child.lineno,
                    decorators=tuple(_name(d) for d in child.decorator_list),
                    bases=tuple(_name(b) for b in child.bases) if is_class else (),
                    class_name=child.name if is_class else class_name,
                    node=child,
                ))
                walk(child, qualname, child.name if is_class else class_name)

    walk(tree, "", "")
    return found


def functions(source: str, filename: str) -> list[Unit]:
    return [unit for unit in units(source, filename) if unit.kind == "function"]


def classes(source: str, filename: str) -> list[Unit]:
    return [unit for unit in units(source, filename) if unit.kind == "class"]


def enclosing(all_units: Iterable[Unit], line: int, kind: str = "function") -> Unit | None:
    """Innermost unit of ``kind`` containing ``line``, or None."""
    best: Unit | None = None
    for unit in all_units:
        if unit.kind == kind and unit.covers(line):
            if best is None or unit.length < best.length:
                best = unit
    return best


def touched_units(case, kind: str = "function") -> Iterator[tuple[str, Unit]]:
    """Units of every changed file that the pull request actually edits.

    A unit counts as touched when the diff adds or rewrites any line inside it,
    which is the population an enumerator is allowed to look at under PR-only
    scanning.
    """
    for filename, source in sorted(case.head_files.items()):
        added = case.added_lines.get(filename, frozenset())
        if not added:
            continue
        for unit in units(source, filename):
            if unit.kind != kind:
                continue
            if any(unit.covers(line) for line in added):
                yield filename, unit


def statements(source: str, filename: str) -> list[ast.stmt]:
    tree = parse(source)
    return [] if tree is None else [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
