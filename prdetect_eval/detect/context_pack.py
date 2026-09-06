"""Stage [3]: turn candidate sites into the text the model is shown.

One pack is one model call. The unit is a *file*, not a site: the sibling table
is per-file and `authz` judgment is comparative -- an endpoint with no
`permission_classes` means one thing beside five that declare it and another in a
file where none do. Packing a file at once also shows overlapping regions once
instead of per site, which is where the token budget actually goes.

Two properties are load-bearing.

*Real line numbers.* The model answers with `file:line`, so the code block is
printed with the file's own numbering. A block that renumbered from one would
make every answer wrong in a way no metric could distinguish from bad judgment.

*Diff marks.* Only lines the pull request touched carry `+`. Bitbucket can annotate
nothing else, so a finding on an unmarked line is unreportable in production even
when it is true.

Nothing here calls a model, so the prompt budget of a whole corpus is measurable
without a GPU -- which is what PLAN.md section 9, gap 8 asks for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from candidates.context import Tree
from schema import Candidate, Case, Span

from . import prompt as prompt_module

KIND_DESCRIPTIONS = {
    "view_class": "endpoint class",
    "orm_read": "database read",
    "request_derived": "value taken from the request",
    "unused_scope_parameter": "scoping parameter never read",
}


@dataclass(frozen=True)
class Pack:
    """One model call: a stable system prefix plus one file's worth of sites."""

    case_id: str
    filename: str
    candidates: tuple[Candidate, ...]
    system: str
    user: str
    shown_lines: int

    @property
    def estimated_tokens(self) -> int:
        """Rough size, for budgeting before a server exists.

        Ollama returns ``prompt_eval_count`` once it runs, and that number
        replaces this one in the manifest; four characters per token is only
        close enough to decide a context window.
        """
        return (len(self.system) + len(self.user)) // 4

    @property
    def focus_lines(self) -> tuple[int, ...]:
        return tuple(sorted({candidate.focus.start_line for candidate in self.candidates}))


def _merge(spans: Sequence[Span]) -> list[tuple[int, int]]:
    """Collapse overlapping or touching regions into display blocks."""
    ordered = sorted((span.start_line, span.end_line) for span in spans)
    blocks: list[tuple[int, int]] = []
    for start, end in ordered:
        if blocks and start <= blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    return blocks


def _decorator_start(tree: Tree, filename: str, line: int) -> int:
    """First decorator line of a unit defined at ``line``, or ``line`` itself.

    ``ast`` puts a decorated function's ``lineno`` on the ``def``, so a region
    taken from a unit span starts *below* ``@permission_classes(...)``. For this
    detector that is the single most important line in the file.
    """
    best = line
    for unit in tree.units(filename):
        node = getattr(unit, "node", None)
        decorators = getattr(node, "decorator_list", None)
        if unit.def_line == line and decorators:
            best = min(best, min(decorator.lineno for decorator in decorators))
    return best


def _code_block(case: Case, tree: Tree, filename: str, blocks: Sequence[tuple[int, int]]) -> tuple[list[str], int]:
    lines = tree.lines(filename)
    added = case.added_lines.get(filename, frozenset())
    rendered: list[str] = []
    shown = 0
    for index, (start, end) in enumerate(blocks):
        if index:
            rendered.append("       |")
            rendered.append("       | ...")
            rendered.append("       |")
        for number in range(start, min(end, len(lines)) + 1):
            mark = "+" if number in added else " "
            rendered.append(f"{number:5d} {mark} | {lines[number - 1]}")
            shown += 1
    return rendered, shown


def _sibling_rows(candidates: Sequence[Candidate]) -> list[dict]:
    for candidate in candidates:
        rows = candidate.evidence.get("siblings")
        if rows:
            return rows
    return []


def _sibling_table(rows: Sequence[dict]) -> list[str]:
    if not rows:
        return []
    out = [
        "## Endpoint classes declared in this file",
        "",
        "| class | line | permission_classes | authentication_classes | scoped get_queryset |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row.get('class', '')} | {row.get('line', '')} "
            f"| {'yes' if row.get('permission_classes') else 'no'} "
            f"| {'yes' if row.get('authentication_classes') else 'no'} "
            f"| {'yes' if row.get('scoped_queryset') else 'no'} |"
        )
    return out + [""]


def _describe(candidate: Candidate) -> str:
    evidence = candidate.evidence
    kind = KIND_DESCRIPTIONS.get(evidence.get("kind", ""), evidence.get("kind", "site"))
    where = f" in `{candidate.symbol}`" if candidate.symbol else ""
    detail = ""
    if evidence.get("kind") == "orm_read":
        detail = "; organization predicate " + ("present" if evidence.get("scoped") else "absent")
    elif evidence.get("kind") == "view_class":
        detail = "; permission_classes " + ("declared" if evidence.get("permission_classes") else "not declared")
    elif evidence.get("kind") == "request_derived":
        detail = f"; reads `request.{evidence.get('source', '')}`"
    elif evidence.get("kind") == "unused_scope_parameter":
        detail = "; unread parameters: " + ", ".join(evidence.get("parameters", []))
    return f"- line {candidate.focus.start_line}{where}: {kind}{detail}"


def build(case: Case, tree: Tree, candidates: Sequence[Candidate], filename: str) -> Pack:
    ordered = sorted(candidates, key=lambda candidate: (candidate.focus.start_line, candidate.detector))
    spans = [candidate.region for candidate in ordered]
    blocks = _merge(spans)
    blocks = [(_decorator_start(tree, filename, start), end) for start, end in blocks]
    blocks = _merge([Span(filename, start, end) for start, end in blocks])
    code, shown = _code_block(case, tree, filename, blocks)

    body = [
        "# Pull request",
        "",
        case.pr_title.strip() or "(no title)",
        "",
        case.pr_description.strip() or "(no description)",
        "",
        f"# FILE {filename}",
        "",
    ]
    body += _sibling_table(_sibling_rows(ordered))
    body += ["## Sites flagged for review", ""]
    body += [_describe(candidate) for candidate in ordered]
    body += [
        "",
        "These sites are where tenant scoping *could* apply. They are not accusations; "
        "most are correct.",
        "",
        f"## CODE {filename}",
        "",
        "```",
    ]
    body += code
    body += ["```", ""]

    return Pack(
        case_id=case.case_id, filename=filename, candidates=tuple(ordered),
        system=prompt_module.SYSTEM, user="\n".join(body), shown_lines=shown,
    )


def packs_for_case(case: Case, candidates: Iterable[Candidate]) -> list[Pack]:
    """One pack per file that has at least one candidate."""
    tree = Tree(case)
    by_file: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_file.setdefault(candidate.region.file, []).append(candidate)
    return [build(case, tree, group, filename) for filename, group in sorted(by_file.items())]
