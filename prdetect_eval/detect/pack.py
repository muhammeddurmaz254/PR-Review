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

from schema import Case

from . import prompt as prompt_module

# `@@ -old,n +new,m @@ enclosing context`. Only the new-side start is needed:
# it is the line number the reviewer sees, and the one a label refers to.
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$")
COMMIT = re.compile(r"^# commit ([0-9a-f]{7,40}) ?(.*)$")


@dataclass(frozen=True)
class Pack:
    """One model call."""

    case_id: str
    system: str
    user: str
    shown_lines: int

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


def build(case: Case, dataset: str, version: str = prompt_module.PROMPT_VERSION) -> Pack:
    body = [
        "# Pull request",
        "",
        case.pr_title.strip() or "(no title)",
    ]
    if case.pr_description.strip():
        body += ["", case.pr_description.strip()[:1500]]
    body += [""]

    shown = 0
    if case.head_files:
        for filename in sorted(case.head_files):
            added = case.added_lines.get(filename, frozenset())
            code, count = _numbered(case.head_files[filename], added)
            shown += count
            body += [f"# FILE {filename}", "", "```"] + code + ["```", ""]
    else:
        # No checkout available: the diff is the whole of the evidence.
        rendered, shown = render_diff(case.diff)
        body += ["Only the lines this pull request changed are available; the "
                 "rest of each file is not. Line numbers are the file's own."
                 if rendered else "No code is available for this pull request.",
                 ""] + rendered

    return Pack(
        case_id=case.case_id,
        system=prompt_module.system(dataset, version),
        user="\n".join(body),
        shown_lines=shown,
    )
