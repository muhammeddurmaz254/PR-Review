"""Stage [3c]: the rules a repository writes down about the names a change touches.

D29 opened stock_bench and missed all five of its defects that break a written
convention. `stock-01` compares against `level.on_hand`; `docs/conventions.md`
says in so many words that anything taking stock away compares against
`available()`, and that comparing against `on_hand` hands the same units to two
orders. The detector never saw that file: its pack is the changed files, and
the facts section rendered empty.

This is not the precedent clause that D19 removed. That clause inferred a rule
from how sibling code behaves, so a repository with no siblings had no
defects. Here the rule is one the repository states; a repository that states
none gets nothing added and reviews exactly as before.

Three close relatives were measured in this project and lost, and the design
answers them one by one:

* the whole repository in the prompt (-0.117 F1) -- so only passages that name
  something the change touches, never whole files;
* a targeted document of about 750 tokens (worse) -- so a hard cap below that;
* a rule list, which sent the model hunting (-0.067 F1) -- so the section says
  in its own words that most changes break none of these, and nothing in it is
  phrased as a check to run.

Selection is lexical and deterministic: code identifiers on the lines the
change added or deleted, matched against the identifiers a documentation
paragraph writes in backticks, most matches first.

MEASURED (D30) and REJECTED as a default. Same chain otherwise, location
rung (right file, within ten lines, type ignored), 0.8 floor, each corpus
against a baseline at the same context window:

    stock_dev   10/3/12 -> 6/2/16   F1 0.571 -> 0.400
    zincir_dev  11/2/5  -> 11/0/5   F1 0.759 -> 0.815
    halka       42/8/6  -> 43/8/5   F1 0.857 -> 0.869
    demo_repo   unchanged by construction: no documentation, identical packs
    pooled      78/14/25 -> 75/11/28   F1 0.800 -> 0.794, precision 0.848 ->
                0.872, recall 0.757 -> 0.728

It recovered one of stock's five convention defects (`stock-19`) and cost the
detector its voice everywhere else: stock's claims fell from 22 to 10, and
four of the five true findings it lost -- an off-by-one, a substring identity
test, a dropped assertion, a tautological one -- were simply not reported,
and none of them breaks a written rule. The fifth, `stock-02`, is the same
claim at the same confidence judged the other way by the verifier, which never
sees this section: run-to-run variance, not the treatment.

So the passages did not send the model hunting, which is how the rule list
lost. They quietened it. Whether that is the passages or the sentence telling
it most changes break none of them is not separated here, and is the one
question left open. The selection stayed broad after tightening -- a section
on 19 of stock's 20 clean cases and 52 of halka's 61 -- which is the other
reading of the same result.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from typing import Sequence

from schema import Case
from . import pack as pack_module

DOC_FILE = re.compile(r"(^|/)(docs?/.*\.(md|rst|txt)|CONTRIBUTING[^/]*|CONVENTIONS[^/]*|"
                      r"ARCHITECTURE[^/]*)$", re.I)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
# Names that occur in almost every Python line and so select everything.
COMMON = frozenset({*keyword.kwlist, "self", "None", "True", "False", "print", "range", "list",
                    "dict", "tuple", "str", "int", "float", "bool", "len", "super", "cls",
                    "object", "type", "isinstance", "value", "values", "items", "name", "data",
                    "result", "args", "kwargs", "test", "tests", "setUp", "assertEqual",
                    "assertTrue", "assertFalse", "assertIn", "assertRaises", "return", "import"})

BUDGET_CHARS = 2800   # about 700 tokens: under the targeted document that lost


@dataclass(frozen=True)
class Passage:
    file: str
    heading: str
    text: str
    names: tuple[str, ...]


def _code_names(source: str) -> dict[int, set[str]]:
    """NAME tokens per line -- no comments, no strings, no prose."""
    import io
    import tokenize
    out: dict[int, set[str]] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.NAME:
                out.setdefault(token.start[0], set()).add(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


STRIP = re.compile(r"""(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""")


def touched_names(case: Case) -> set[str]:
    """Code identifiers on the lines this change added or deleted.

    The first version read every word on those lines and matched it against
    every word of the docs. The separation test caught it before a run: the
    section filled on 54 of halka's 61 clean cases, selected by `that`,
    `nothing`, `every` from the docs' prose and by the Turkish words of the
    change's own comments. Only what the tokenizer calls a NAME counts now.
    """
    names: set[str] = set()
    removals = pack_module.removed_lines(case)
    for filename, source in case.head_files.items():
        if not pack_module.is_code(filename):
            continue
        by_line = _code_names(source)
        for number in case.added_lines.get(filename, frozenset()):
            names |= by_line.get(number, set())
        for _, text in removals.get(filename, ()):
            names.update(IDENTIFIER.findall(STRIP.sub("", text)))
    return {name for name in names if name not in COMMON and len(name) >= 4}


BACKTICKED = re.compile(r"`([^`]+)`")


def _named_in(text: str) -> set[str]:
    """Identifiers a documentation passage names AS CODE, in backticks.

    A document talks about `on_hand` when it writes it as code; the same letters
    in a sentence are a word. This is what separates a rule about a symbol from
    prose that happens to share a token with it.
    """
    found: set[str] = set()
    for span in BACKTICKED.findall(text):
        found.update(IDENTIFIER.findall(span))
    return found


def _paragraphs(source: str) -> list[tuple[str, str]]:
    """(nearest heading, paragraph) for every blank-line separated block."""
    out, heading, block = [], "", []
    for line in source.splitlines() + [""]:
        if line.startswith("#"):
            if block:
                out.append((heading, "\n".join(block).strip()))
                block = []
            heading = line.lstrip("#").strip()
        elif line.strip():
            block.append(line)
        elif block:
            out.append((heading, "\n".join(block).strip()))
            block = []
    return out


def collect(case: Case, budget: int = BUDGET_CHARS) -> list[Passage]:
    names = touched_names(case)
    if not names:
        return []
    files = {**(case.context_files or {}), **case.head_files}
    scored = []
    for filename in sorted(files):
        if not DOC_FILE.search(filename):
            continue
        for order, (heading, text) in enumerate(_paragraphs(files[filename])):
            hits = tuple(sorted(names & _named_in(text)))
            if hits:
                scored.append((-len(hits), filename, order, Passage(filename, heading, text, hits)))
    chosen, used = [], 0
    for _, _, _, passage in sorted(scored, key=lambda row: row[:3]):
        if used + len(passage.text) > budget:
            continue
        chosen.append(passage)
        used += len(passage.text)
    return sorted(chosen, key=lambda p: (p.file, list(files).index(p.file) if p.file in files else 0))


def render(passages: Sequence[Passage]) -> list[str]:
    if not passages:
        return []
    body = ["# WHAT THIS REPOSITORY WRITES DOWN", "",
            "Quoted from the repository's own documentation, because each passage mentions a "
            "name this pull request touches. They say what the repository expects of its code. "
            "Most changes break none of them; say so by reporting nothing about them.", ""]
    for passage in passages:
        where = f"{passage.file}" + (f" -- {passage.heading}" if passage.heading else "")
        body += [f"From `{where}`:", "", passage.text, ""]
    return body
