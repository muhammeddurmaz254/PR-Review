"""Stage [3b]: answer the questions a reader cannot answer by reading.

Some defects turn on a fact about the whole repository rather than on anything
visible in the change: whether a symbol this pull request defines is referenced
anywhere, whether a constant it introduces already exists under another name,
what the neighbours of a function it adds are called. A reviewer answers these
by searching. A model cannot, and the two ways of helping it were measured and
both lost -- the whole repository in the prompt cost 0.117 of F1 and a targeted
750-token document cost more.

So nothing here shows code. Each entry is the *answer* to one such question, in
a line, computed with `ast` over the revision the pull request produces. That
distinction is the whole design: retrieval hands the model something to read and
dilutes what it was reading; a fact hands it something already settled.

Three rules keep it from becoming the checklist that backfired:

* Only symbols the change itself defines are asked about. A fact about code the
  pull request never touches is a suggestion to go hunting.
* Nothing here judges. "Referenced nowhere" is a count, not an accusation; the
  model decides whether it means anything.
* **A fact that does not discriminate is not stated at all.** Three kinds were
  written and two were deleted before they cost a run: counting a name's
  same-prefix siblings fired on 33 defective cases and 38 clean ones, and
  reporting any unreferenced symbol fired on 32 and 41. Both are true and both
  are noise -- a newly added endpoint is naturally referenced nowhere yet, being
  reached through a URL map. True-but-undiscriminating material is exactly what
  a rule list is, and a rule list cost 0.067 of F1 when it was tried.

What survives is narrow on purpose. Each remaining fact fires on about one case
in a hundred, which is the point: it says something when there is something to
say and is silent otherwise.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Sequence

from schema import Case


@dataclass(frozen=True)
class Fact:
    """One settled question about the repository, and where it came from."""

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
    """Top-level names the changed files define, with the node that defines them.

    Only files the pull request touches, and only definitions -- a name this
    change did not introduce is somebody else's business.
    """
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

    Uses, not definitions: a function that only defines itself is referenced
    nowhere. Attribute access on another object is not counted, which is why the
    statement says how it was counted -- a dynamic call or a name reached through
    getattr is invisible here and "not found this way" is not "unused".
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
    """Other module-level names assigned the same literal, anywhere in the repo."""
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
    """Every settled question worth stating about this change's own symbols."""
    if not case.head_files:
        return []
    facts: list[Fact] = []
    seen: set[str] = set()
    for name, node in _defined_here(case):
        if name in seen or name.startswith("__"):
            continue
        seen.add(name)

        # Only a module-private name. A public one that nothing references may
        # simply be new, or reached through a router, a registry or a decorator
        # that no name lookup can see -- measured, reporting those fired on more
        # clean cases than defective ones. The leading underscore is the
        # language's own statement that nothing outside should be calling it,
        # which is what makes the count mean something.
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


# --- facts about one claim: two were written for the verifier, measured, deleted
#
# A copy count for duplication claims -- the challenger had said it could not
# see the function a claim called the original. Separation test, one row per
# function: the two labelled real copies shared 1.0 and 0.71 of their
# statements, names aside, with their closest other function; the clean-twin
# false alarms shared 0.83, 0.80, 0.0 and 0.0. The two it was written for are
# more alike than one of the real copies. Given to the verifier, the count
# would have argued for them. Deleted before it cost a run.
#
# The other side of a deletion: what this change deleted the definition of, and
# what still reads it. Given to the verifier over the best chain it reached
# five zincir claims and one halka claim and moved none: 8/3/8 and 44/9/4
# unchanged. The claim it was written for, dlq-01-k1, was refuted again, and
# rightly -- the key is still there; what is missing is the table entry it now
# reads from. The separation test, with definitions read off the revision
# before the change rather than guessed from a deleted line, is no better: it
# fires on both twins of dlq-01 and authz-01, on clean-ups of dead code, and
# its "still read" half on one defective case already found (ckpt-01) and one
# clean refactor (demo_repo's temiz-yeniden-duzenleme, whose dict keys are
# built another way). Its first version took a keyword argument and a local
# variable for definitions and `response.json()` for a read; a count that has
# to be defended line by line is not a settled answer. Deleted.
