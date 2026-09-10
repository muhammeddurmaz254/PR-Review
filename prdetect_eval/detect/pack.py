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
from fnmatch import fnmatch
from typing import Sequence

from schema import Case

from . import prompt as prompt_module

# `@@ -old,n +new,m @@ enclosing context`. Only the new-side start is needed:
# it is the line number the reviewer sees, and the one a label refers to.
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$")

# Only source is reviewed. A reviewer of a Python service reads Python; prose and
# manifests are a different job with different evidence, and the two labels that
# live in one -- a package moved between requirements files, a password left in a
# README -- are marked out of scope by the exporters rather than left reachable
# in a pack that no longer shows them.
CODE_SUFFIXES = (".py",)


def is_code(filename: str) -> bool:
    return filename.endswith(CODE_SUFFIXES)
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
        if not is_code(hunk.filename):
            continue
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


def shown_lines(case: Case, context: Sequence[str] = ()) -> tuple[dict[str, dict[int, list[str]]], dict[str, list[str]]]:
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
        sources = {name: text for name, text in case.head_files.items() if is_code(name)}
        if context:
            # The filter compares a quote against what was printed; leaving the
            # unchanged files out would reject every quote taken from them.
            sources |= {name: text for name, text in case.context_files.items()
                        if any(fnmatch(name, pattern) for pattern in context)}
        for filename, source in sources.items():
            rows, _ = _numbered(source, case.added_lines.get(filename, frozenset()))
            numbered[filename] = {int(row[:5]): [row.split("| ", 1)[-1]] for row in rows}
        return numbered, deleted
    for hunk in emitted_hunks(case.diff):
        if not is_code(hunk.filename):
            continue
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
          context: Sequence[str] = ()) -> Pack:
    """The whole pull request in one call."""
    return split(case, dataset, version, max_lines=0, context=context)[0]


def _repo_section(case: Case, patterns: Sequence[str] = ("*",)) -> list[str]:
    """Unchanged files from the repository at head, after the diff.

    Carrying *everything* was measured and lost: rung 1 on halka_bench scored
    worse than the diff alone on every metric, gaining the two defect types that
    need the repository and losing four that were visible in the change. A fixed
    attention budget spread over twenty-seven thousand tokens stops seeing the
    diff. So the whole repository is the ceiling a retrieval policy is bounded
    by, not a policy -- and since the ceiling is below the floor, a policy can
    only win by being narrow. ``patterns`` is how narrow.
    """
    chosen = {name: text for name, text in case.context_files.items()
              if is_code(name) and any(fnmatch(name, pattern) for pattern in patterns)}
    if not chosen:
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
    for filename in sorted(chosen):
        rows, _ = _numbered(chosen[filename], frozenset())
        body += [f"# FILE {filename}   (unchanged)", "", "```"] + rows + ["```", ""]
    return body


def split(case: Case, dataset: str, version: str = prompt_module.PROMPT_VERSION,
          max_lines: int = 0, context: Sequence[str] = ()) -> list[Pack]:
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
        body += ["# WHAT THIS PULL REQUEST CHANGED", ""] if context else []
        for filename in sorted(name for name in case.head_files if is_code(name)):
            added = case.added_lines.get(filename, frozenset())
            code, count = _numbered(case.head_files[filename], added)
            shown += count
            body += [f"# FILE {filename}", "", "```"] + code + ["```", ""]
        if context:
            body += _repo_section(case, context)
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
