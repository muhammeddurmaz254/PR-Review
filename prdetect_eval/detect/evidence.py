"""Stage [5b]: hold a finding to the operation its own type names.

Stage [5] asks whether the accused line exists. This asks the next question, and
only of the types that admit it: some defect classes are named after an
operation. An SSRF is a fetch. An SQL injection is untrusted text inside query
text. A finding of that kind is a claim about what one statement *does*, so if
the statement does not do it, the claim is about a callee that was never shown
and the pull request cannot support it.

Nothing here is a taste judgement and nothing here reads a label. The rule is
the same discipline as the quote gate one step up: a report must point at the
thing it accuses.

Three cautions are built in, because the obvious version of this filter is
wrong in three measured ways.

*The unit is the statement, not the line.* A call wrapped over four lines is one
operation; a report anchored on its second argument is still anchored on the
call. Under a line-scoped test prompt v7's report on ``ssrf-02-kusurlu``, which
named the argument line of a ``requests.get(`` that began one line above, was
dropped as a true positive. Statement scope keeps it.

*Except where the argument position is the defect.* ``sql_injection`` is exactly
the case: ``raw_query("... ILIKE %s", (org_id, "%%%s%%" % terim))`` interpolates
a user string, and is safe, because the interpolation happens in the parameter
tuple and not in the query text. Widening to the statement erases the only thing
that distinguishes it from the injection. So that signature reads the call's
arguments rather than its text.

*A type only gets a signature when its name really is an operation.* In this
corpus ``weak_crypto_primitive`` covers a non-constant-time comparison --
``connector.token_hash == hash_token(token)`` -- which names no primitive at
all. It has no signature, and a type whose defect can live off the accused line
must not be given one.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from schema import Case

from . import contract

# A raw egress call: a third-party HTTP client invoked directly. A project-local
# wrapper is deliberately absent -- the point is not that some helper is known to
# be safe, but that a report naming a line which only calls a wrapper is accusing
# the wrapper's body, and the wrapper's body was not shown.
EGRESS = re.compile(
    r"\b(?:requests|httpx|aiohttp|urllib3|urllib\.request|http\.client)\s*\.|"
    r"\burlopen\s*\(|\bcurl\b", re.I)

# Decoders that turn bytes into objects and can be made to do more than that.
# XML parsers belong here: an XXE is an unsafe decode, and any list of
# deserializers written without this corpus in front of it would name them.
DECODE = re.compile(
    r"\b(?:pickle|cPickle|dill|marshal|shelve|jsonpickle)\s*\.|"
    r"\byaml\.(?:unsafe_)?load\s*\(|\b(?:eval|exec)\s*\(|"
    r"\b(?:ElementTree|etree|minidom|expat|xmlrpc)\b|"
    r"\b(?:fromstring|parseString)\s*\(", re.I)

SQL_KEYWORD = re.compile(
    r"\b(?:select|insert|update|delete|where|from|join|union|ilike|like|"
    r"order\s+by|group\s+by)\b", re.I)


def _first_party_roots(case: Case, filename: str) -> set[str]:
    """The import roots that belong to the project under review.

    Read from the paths in the pull request, not from a list anyone has to keep:
    a reviewer looking at ``halka/integrations/client.py`` knows ``halka`` names
    this project without being told, and so does a reviewer of ``src/app/views.py``.
    """
    roots = set()
    for name in [filename, *case.head_files]:
        parts = [part for part in name.split("/") if part not in {"", ".", "src", "lib"}]
        if len(parts) > 1 and not parts[0].endswith(".py"):
            roots.add(parts[0])
    return roots


def _only_first_party_calls(case: Case, filename: str, node: ast.stmt) -> bool:
    """Whether every call in this statement is one of the project's own functions.

    This is the portable half of the fetch and decode signatures, and it asks a
    better question than "was a known library named". What a reviewer can settle
    from one statement is not which library this is, but whether the operation is
    visible at all. ``fetch_url(icon_url)`` is a bare name imported from this
    project: whatever it does happens in a body that was never shown, so the
    accusation is misplaced at this line -- and that holds without deciding that
    the wrapper is safe.

    Everything else keeps the finding, which is the direction that has to fail
    open. Measured against the module-prefix list this replaces:
    ``self.session.get(url)``, ``opener.open(url)``, ``await self.client.get(url)``
    and an aliased ``rq.get(url)`` were all missed, and their findings would have
    been deleted. A pooled ``requests.Session`` or ``httpx.Client`` held on
    ``self`` is the dominant idiom in production code, not the exception, so a
    filter that only recognises module-level calls is a filter that deletes real
    findings in every repository but this one.
    """
    try:
        module = ast.parse("\n".join(case.source_lines(filename)))
    except (SyntaxError, ValueError):
        return False
    roots = _first_party_roots(case, filename)
    local: set[str] = set()
    for item in ast.walk(module):
        if isinstance(item, ast.ImportFrom):
            # A relative import is this project by construction.
            if item.level or (item.module or "").split(".")[0] in roots:
                local.update(alias.asname or alias.name for alias in item.names)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local.add(item.name)

    calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
    if not calls:
        return False
    return all(isinstance(call.func, ast.Name) and call.func.id in local for call in calls)


def _statement(case: Case, filename: str, line: int, end_line: int) -> str:
    """The whole statement the report sits in, or the raw lines if it will not parse."""
    source = case.source_lines(filename)
    raw = "\n".join(source[max(0, line - 1):max(end_line, line)])
    node = _statement_node(case, filename, line)
    if node is None:
        return raw
    start, stop = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
    return "\n".join(source[start - 1:max(stop, end_line)])


def _statement_node(case: Case, filename: str, line: int) -> ast.stmt | None:
    """The innermost simple statement covering ``line``.

    Definitions and modules are skipped: a report inside a function would
    otherwise widen to the whole function and every signature would match.
    """
    try:
        tree = ast.parse("\n".join(case.source_lines(filename)))
    except (SyntaxError, ValueError):
        return None
    best: ast.stmt | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stop = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= line <= stop:
            if best is None or stop - node.lineno < (getattr(best, "end_lineno", best.lineno) or best.lineno) - best.lineno:
                best = node
    return best


def _dynamic(node: ast.AST) -> bool:
    """Whether an expression splices a runtime value into a string."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.JoinedStr):
            return any(isinstance(part, ast.FormattedValue) for part in inner.values)
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, (ast.Mod, ast.Add)):
            return True
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) \
                and inner.func.attr in {"format", "join"}:
            return True
    return False


def _sql_text_is_built(case: Case, filename: str, line: int, text: str) -> bool:
    """Whether a runtime value is spliced into an argument that carries SQL.

    The parameter tuple of a placeholder query holds user data by design; that is
    what a placeholder query is for. Only an argument that is itself SQL -- it
    names SQL keywords -- and is assembled at runtime is an injection.
    """
    node = _statement_node(case, filename, line)
    if node is None:
        # No tree: fall back to the coarse reading rather than clearing the
        # finding on a parse failure.
        return bool(SQL_KEYWORD.search(text) and _INTERP.search(text))
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        for argument in list(call.args) + [kw.value for kw in call.keywords]:
            try:
                rendered = ast.unparse(argument)
            except Exception:  # pragma: no cover - unparse is total on 3.9+
                continue
            if SQL_KEYWORD.search(rendered) and _dynamic(argument):
                return True
    return False


_INTERP = re.compile(r"%\s*[\(\w\"']|\.format\s*\(|\bf['\"]|\{\w*\}")


@dataclass(frozen=True)
class Signature:
    """What a type's name promises the accused statement will contain."""

    detail: str
    holds: Callable[[Case, str, int, str], bool]


def _visible_operation(pattern: re.Pattern[str]):
    """Keep the finding when the statement names a known call, or is not all ours.

    Two ways to survive and one way to be dropped. The pattern is a positive
    override for the idioms everyone recognises; the first-party test is what
    carries the rule to a repository whose spelling nobody listed.
    """
    def holds(case: Case, filename: str, line: int, text: str) -> bool:
        if pattern.search(text):
            return True
        node = _statement_node(case, filename, line)
        if node is None:
            return True
        return not _only_first_party_calls(case, filename, node)
    return holds


SIGNATURES: dict[str, Signature] = {
    "ssrf_unvalidated_fetch": Signature(
        "the accused statement only calls this project's own functions",
        _visible_operation(EGRESS)),
    "unsafe_deserialization": Signature(
        "the accused statement only calls this project's own functions",
        _visible_operation(DECODE)),
    "sql_injection": Signature(
        "no runtime value inside the query text",
        _sql_text_is_built),
}


@dataclass(frozen=True)
class Decision:
    report: contract.Report
    verdict: str
    detail: str = ""

    @property
    def kept(self) -> bool:
        return self.verdict != "off-operation"


def resolve(reports: Sequence[contract.Report], case: Case) -> list[Decision]:
    """One decision per report, in order. Types without a signature pass."""
    decisions: list[Decision] = []
    for report in reports:
        signature = SIGNATURES.get(report.type)
        if signature is None:
            decisions.append(Decision(report, "no-signature"))
            continue
        if report.file not in case.head_files:
            # The filter reads source; without it the report is left alone and
            # pays for itself in the false-alarm column instead.
            decisions.append(Decision(report, "unchecked", "file not in the pull request"))
            continue
        text = _statement(case, report.file, report.line, report.line)
        if signature.holds(case, report.file, report.line, text):
            decisions.append(Decision(report, "on-operation"))
        else:
            decisions.append(Decision(report, "off-operation", signature.detail))
    return decisions


def apply(decisions: Sequence[Decision]) -> list[contract.Report]:
    return [d.report for d in decisions if d.kept]
