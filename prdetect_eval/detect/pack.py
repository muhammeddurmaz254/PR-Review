"""Turn one case into the text the model reads. One call per pull request.

There is no candidate enumeration stage. It was measured away: a demo_repo case
changes 1.4 files averaging 356 tokens of content, and a SWRBench case is a
single hunk of 88. Selecting sites inside that would add a stage that can only
lose findings -- and eight of demo_repo's eighteen defects exist only in the
relationship between two changed files, which per-file packing would make
unreachable by construction.

Two shapes are handled, and both print real file line numbers. A case with file
contents is printed whole, with a `+` on the lines the pull request touched. A
case without them -- SWRBench keeps only diffs, since the alternative is cloning
eleven upstream repositories -- has its hunks parsed and renumbered from the
`@@` headers, so a hunk reads like an excerpt of the file rather than like a
patch the model has to count through.

That renumbering is not cosmetic. The response contract asks for a file path "as
printed in a FILE header" and a line "printed in the code"; against a raw diff
neither existed, so the instruction was unsatisfiable for half the corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from schema import Case

from . import prompt as prompt_module

# `@@ -old,n +new,m @@ enclosing context`. Only the new-side start is needed:
# it is the line number the reviewer sees, and the one a label refers to.
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$")
COMMIT = re.compile(r"^# commit ([0-9a-f]{7,40}) ?(.*)$")


@dataclass(frozen=True)
class Pack:
    """One model call. A large pull request makes several."""

    case_id: str
    system: str
    user: str
    shown_lines: int
    part: int = 1
    parts: int = 1

    @property
    def estimated_tokens(self) -> int:
        """Rough size for budgeting before a server exists.

        Ollama reports ``prompt_eval_count`` once it runs and that replaces this
        in the manifest; four characters per token is only close enough to
        choose a context window.
        """
        return (len(self.system) + len(self.user)) // 4


def _numbered(source: str, added: frozenset[int]) -> tuple[list[str], int]:
    lines = source.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    out = [f"{number:5d} {'+' if number in added else ' '} | {text}"
           for number, text in enumerate(lines, start=1)]
    return out, len(out)


@dataclass(frozen=True)
class Hunk:
    """One `@@` block, renumbered onto the file it patches."""

    filename: str
    commit: str
    heading: str
    start: int
    end: int
    lines: list[str]


def parse_diff(text: str) -> list[Hunk]:
    """Read the concatenated-commit diff the SWRBench builder writes.

    The shape is regular across all 108 commit diffs in the sample: an optional
    `# commit` marker, a bare path on its own line, then `@@` hunks. Every line
    that leaves a hunk was measured to be a path, so a context line -- which
    always carries its leading space -- can never be mistaken for one.
    """
    hunks: list[Hunk] = []
    filename = commit = heading = ""
    number = 0
    body: list[str] = []

    def flush() -> None:
        # A hunk's trailing blank context line is a diff artefact, not code.
        while body and body[-1].endswith("| "):
            body.pop()
        numbers = [int(row[:5]) for row in body if row[:5].strip()]
        if numbers:
            hunks.append(Hunk(filename, commit, heading,
                              min(numbers), max(numbers), list(body)))
        body.clear()

    inside = False
    for line in text.split("\n"):
        match = HUNK.match(line)
        if match:
            flush()
            inside = True
            number = int(match.group(1))
            heading = match.group(2).strip()
            continue
        if inside:
            mark = line[:1]
            if mark == "+":
                body.append(f"{number:5d} + | {line[1:]}")
                number += 1
                continue
            if mark == "-":
                # A removed line has no place in the new file; it is still shown,
                # because a defect can be exactly what the change deleted.
                body.append(f"{'':5s} - | {line[1:]}")
                continue
            if mark == " " or line == "":
                body.append(f"{number:5d}   | {line[1:]}")
                number += 1
                continue
            flush()
            inside = False
        if not line.strip():
            continue
        marker = COMMIT.match(line)
        if marker:
            commit = f"{marker.group(1)[:7]} {marker.group(2)}".strip()
        else:
            filename = line.strip()
    flush()
    return hunks


def emitted_hunks(text: str, max_lines: int = 1200) -> list[Hunk]:
    """The hunks the cap actually lets through, so the filter and the prompt agree."""
    kept, shown = [], 0
    for hunk in parse_diff(text):
        if shown >= max_lines:
            break
        kept.append(hunk)
        shown += len(hunk.lines)
    return kept


def render_hunks(hunks: Sequence[Hunk]) -> tuple[list[str], int]:
    """Hunks as fenced excerpts under their file and commit headers."""
    out: list[str] = []
    shown = 0
    previous = None
    for hunk in hunks:
        key = (hunk.filename, hunk.commit)
        if key != previous:
            header = f"# FILE {hunk.filename}"
            if hunk.commit:
                header += f"   (commit {hunk.commit})"
            out += [header, ""]
            previous = key
        where = f"Lines {hunk.start}-{hunk.end}"
        out += [f"{where}, in `{hunk.heading}`" if hunk.heading else where, ""]
        out += ["```"] + hunk.lines + ["```", ""]
        shown += len(hunk.lines)
    return out, shown


def render_diff(text: str, max_lines: int = 1200) -> tuple[list[str], int]:
    """The hunks as fenced excerpts, newest-style headers, capped.

    The cap is a safety valve rather than a budget: the largest pull request in
    the sample renders in 1128 lines, so nothing is dropped today, and a corpus
    that outgrows it says so in the prompt instead of silently losing evidence.
    """
    hunks = parse_diff(text)
    out: list[str] = []
    shown = 0
    previous = None
    for index, hunk in enumerate(hunks):
        if shown >= max_lines:
            out += [f"({len(hunks) - index} further hunks are not shown here.)", ""]
            break
        key = (hunk.filename, hunk.commit)
        if key != previous:
            header = f"# FILE {hunk.filename}"
            if hunk.commit:
                header += f"   (commit {hunk.commit})"
            out += [header, ""]
            previous = key
        where = f"Lines {hunk.start}-{hunk.end}"
        out += [f"{where}, in `{hunk.heading}`" if hunk.heading else where, ""]
        out += ["```"] + hunk.lines + ["```", ""]
        shown += len(hunk.lines)
    return out, shown


def shown_lines(case: Case, with_repo: bool = False) -> tuple[dict[str, dict[int, list[str]]], dict[str, list[str]]]:
    """Exactly what the pack prints, as ``(numbered, deleted)``.

    ``numbered`` maps file to line number to the texts printed at it. A list,
    not a string: a file touched by two commits is printed twice with each
    commit's own numbering, so one number can carry two different lines, and
    collapsing them would let the filter reject a quote the model was shown.

    ``deleted`` holds the removed lines, which have no number in the new file
    and so can be quoted but never anchored -- a defect that *is* the deletion
    has nothing else to quote.
    """
    numbered: dict[str, dict[int, list[str]]] = {}
    deleted: dict[str, list[str]] = {}
    if case.head_files:
        sources = dict(case.head_files)
        if with_repo:
            # The filter compares a quote against what was printed; leaving the
            # unchanged files out would reject every quote taken from them.
            sources |= case.context_files
        for filename, source in sources.items():
            rows, _ = _numbered(source, case.added_lines.get(filename, frozenset()))
            numbered[filename] = {int(row[:5]): [row.split("| ", 1)[-1]] for row in rows}
        return numbered, deleted
    for hunk in emitted_hunks(case.diff):
        for row in hunk.lines:
            head, _, text = row.partition("| ")
            number = head[:5].strip()
            if number.isdigit():
                numbered.setdefault(hunk.filename, {}).setdefault(int(number), []).append(text)
            else:
                deleted.setdefault(hunk.filename, []).append(text)
    return numbered, deleted


def _preamble(case: Case, part: int, parts: int) -> list[str]:
    body = ["# Pull request", "", case.pr_title.strip() or "(no title)"]
    if case.pr_description.strip():
        body += ["", case.pr_description.strip()[:1500]]
    if parts > 1:
        body += ["", f"This is excerpt {part} of {parts} from this pull request. "
                     f"Review what is here; the other excerpts are being reviewed "
                     f"separately, so do not report a defect you cannot see."]
    return body + [""]


def build(case: Case, dataset: str, version: str = prompt_module.PROMPT_VERSION,
          with_repo: bool = False) -> Pack:
    """The whole pull request in one call."""
    return split(case, dataset, version, max_lines=0, with_repo=with_repo)[0]


def _repo_section(case: Case) -> list[str]:
    """The rest of the repository at head, unchanged, after the diff.

    Two thirds of halka_bench's defects are only legible against a convention its
    base branch establishes somewhere the pull request never opens -- a predicate
    a sibling endpoint uses, a unit contract in a common module. Carrying the
    repository is not a retrieval strategy; it is the ceiling a retrieval
    strategy would be measured against, and it fits here because the repository
    is fifty-nine files.
    """
    if not case.context_files:
        return []
    body = [
        "# THE REST OF THE REPOSITORY",
        "",
        "These files are not changed by this pull request. They are here because "
        "a change is judged against the code around it: the conventions this "
        "repository already follows are stated by these files, not by the diff. "
        "Do not report a defect in them -- report only what this pull request does.",
        "",
    ]
    for filename in sorted(case.context_files):
        rows, _ = _numbered(case.context_files[filename], frozenset())
        body += [f"# FILE {filename}   (unchanged)", "", "```"] + rows + ["```", ""]
    return body


def split(case: Case, dataset: str, version: str = prompt_module.PROMPT_VERSION,
          max_lines: int = 0, with_repo: bool = False) -> list[Pack]:
    """The pull request as one pack, or as several when it is large.

    Measured on SWRBench: the model's output volume tracks the size of the pack
    rather than the number of defects in it. A fifty-one hunk pull request drew
    thirty-two findings and located nothing extra, while half the small ones drew
    an empty answer. Both look like one fixed attention budget spread over
    whatever it is given, so the fix is to give it comparable amounts.

    Whole-file cases are never split. Eight demo_repo defects exist only in the
    relationship between two changed files, and separating them would make those
    unreachable by construction; at the thresholds in use no demo_repo case is
    large enough to split anyway, so the rule costs nothing and removes the risk.
    """
    system = prompt_module.system(dataset, version)

    if case.head_files:
        body = _preamble(case, 1, 1)
        shown = 0
        body += ["# WHAT THIS PULL REQUEST CHANGED", ""] if with_repo else []
        for filename in sorted(case.head_files):
            added = case.added_lines.get(filename, frozenset())
            code, count = _numbered(case.head_files[filename], added)
            shown += count
            body += [f"# FILE {filename}", "", "```"] + code + ["```", ""]
        if with_repo:
            body += _repo_section(case)
        return [Pack(case.case_id, system, "\n".join(body), shown)]

    hunks = emitted_hunks(case.diff)
    if not hunks:
        body = _preamble(case, 1, 1) + ["No code is available for this pull request.", ""]
        return [Pack(case.case_id, system, "\n".join(body), 0)]

    groups: list[list[Hunk]] = [[]]
    running = 0
    for hunk in hunks:
        if max_lines > 0 and groups[-1] and running + len(hunk.lines) > max_lines:
            groups.append([])
            running = 0
        groups[-1].append(hunk)
        running += len(hunk.lines)

    packs = []
    for index, group in enumerate(groups, start=1):
        rendered, shown = render_hunks(group)
        body = _preamble(case, index, len(groups))
        body += ["Only the lines this pull request changed are available; the "
                 "rest of each file is not. Line numbers are the file's own.", ""]
        body += rendered
        packs.append(Pack(case.case_id, system, "\n".join(body), shown, index, len(groups)))
    return packs
