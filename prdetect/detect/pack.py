"""The text the detector reads: one pull request, one call.

The files a pull request changed are printed whole with their real line numbers,
a `+` on each line it added or rewrote and, when asked, the lines it deleted,
printed where they were. Only source is printed. The changed files are printed
together, because some defects exist only between two of them.

`shown_lines` is exactly what a pack printed, so the anchor check compares a quote
with what the model saw; `excerpt` is the window around one claim that the
verifier is shown, numbered the same way.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from prdetect.cases import Case, is_code
from prdetect.detect import facts as facts_module
from prdetect.detect import prompt as prompt_module

# `@@ -old,n +new,m @@`. Only the new-side start is needed: it is the line
# number the reviewer sees.
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$")
DIFF_FILE = re.compile(r"^diff --git a/.+ b/(.+)$")
DIFF_SKIP = ("---", "+++", "index ", "new file", "deleted file",
             "similarity ", "rename ")

DELETIONS_NOTE = ("Lines marked `-` were deleted by this pull request. They "
                  "carry no line number because they are in no file any more; "
                  "they are printed where they used to be.")


@dataclass(frozen=True)
class Pack:
    """One model call."""

    case_id: str
    system: str
    user: str
    shown_lines: int

    @property
    def estimated_tokens(self) -> int:
        """Four characters a token: close enough to choose a context window before a server answers."""
        return (len(self.system) + len(self.user)) // 4


def removed_lines(case: Case) -> dict[str, list[tuple[int, str]]]:
    """Lines the pull request deleted, keyed by file, placed on the new side.

    A defect can be exactly what a change deleted -- a guard, a reset, a key --
    and a whole-file print has no place for a deleted line, so these are printed
    beside it. A line whose content comes back as an added line in the same file
    was moved rather than removed and is left out: a reordered table would
    otherwise read as a list of deletions.

    The number returned is the new-side line the removed text sat in front of. It
    orders the printing and is never a line the model may cite.
    """
    out: dict[str, list[tuple[int, str]]] = {}
    added: dict[str, list[str]] = {}
    filename = ""
    number = 0
    for line in case.diff.split("\n"):
        match = DIFF_FILE.match(line)
        if match:
            filename = match.group(1).strip()
            continue
        hunk = HUNK.match(line)
        if hunk:
            number = int(hunk.group(1))
            continue
        if not filename or line.startswith(DIFF_SKIP):
            continue
        if line.startswith("+"):
            added.setdefault(filename, []).append(line[1:].strip())
            number += 1
        elif line.startswith("-"):
            out.setdefault(filename, []).append((number, line[1:]))
        else:
            number += 1
    kept: dict[str, list[tuple[int, str]]] = {}
    for name, rows in out.items():
        pool = Counter(added.get(name, ()))
        for where, text in rows:
            if pool[text.strip()]:
                pool[text.strip()] -= 1      # moved, not removed
                continue
            kept.setdefault(name, []).append((where, text))
    return kept


def _numbered(source: str, added: frozenset[int]) -> list[str]:
    lines = source.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return [f"{number:5d} {'+' if number in added else ' '} | {text}"
            for number, text in enumerate(lines, start=1)]


def _with_removals(rows: list[str], removals: Sequence[tuple[int, str]]) -> list[str]:
    """Numbered rows with the removed lines printed where they were: blank number, `-` mark."""
    if not removals:
        return rows
    at: dict[int, list[str]] = {}
    for where, text in removals:
        at.setdefault(where, []).append(f"{'':5s} - | {text}")
    out: list[str] = []
    for index, row in enumerate(rows, start=1):
        out += at.pop(index, [])
        out.append(row)
    for where in sorted(at):
        out += at[where]
    return out


def _preamble(case: Case) -> list[str]:
    body = ["# Pull request", "", case.pr_title.strip() or "(no title)"]
    if case.pr_description.strip():
        body += ["", case.pr_description.strip()[:1500]]
    return body + [""]


def build(case: Case, with_facts: bool = False, with_deletions: bool = False) -> Pack:
    """The whole pull request in one pack."""
    body = _preamble(case)
    shown = 0
    removals = removed_lines(case) if with_deletions else {}
    if any(removals.values()):
        body += [DELETIONS_NOTE, ""]
    for filename in sorted(name for name in case.head_files if is_code(name)):
        rows = _numbered(case.head_files[filename], case.added_lines.get(filename, frozenset()))
        # Removals carry no line number, so they are not counted as shown code.
        shown += len(rows)
        rows = _with_removals(rows, removals.get(filename, ()))
        body += [f"# FILE {filename}", "", "```"] + rows + ["```", ""]
    if with_facts:
        body += facts_module.render(facts_module.collect(case))
    return Pack(case.case_id, prompt_module.system(), "\n".join(body), shown)


def shown_lines(case: Case, with_deletions: bool = False) -> tuple[dict[str, dict[int, str]], dict[str, list[str]]]:
    """Exactly what `build` prints, as `(numbered, deleted)`.

    `numbered` maps file to line number to the text printed there. `deleted`
    holds the removed lines, which have no number: they can be quoted but never
    anchored to a line of their own.
    """
    numbered: dict[str, dict[int, str]] = {}
    deleted: dict[str, list[str]] = {}
    removals = removed_lines(case) if with_deletions else {}
    for filename, source in case.head_files.items():
        if not is_code(filename):
            continue
        rows = _numbered(source, case.added_lines.get(filename, frozenset()))
        numbered[filename] = {int(row[:5]): row.split("| ", 1)[-1] for row in rows}
        for _, text in removals.get(filename, ()):
            deleted.setdefault(filename, []).append(text)
    return numbered, deleted


def excerpt(case: Case, filename: str, line: int, radius: int = 12,
            with_deletions: bool = False) -> list[str]:
    """The lines around a claim, numbered and marked exactly as the detector saw them.

    The window is chosen by printed number, not by position: a removal carries no
    number of its own and belongs to the window of the line it sits in front of.
    """
    source = case.head_files.get(filename)
    if source is None:
        return []
    rows = _numbered(source, case.added_lines.get(filename, frozenset()))
    if with_deletions:
        rows = _with_removals(rows, removed_lines(case).get(filename, ()))
    low, high, out, seen = line - radius, line + radius, [], 0
    for row in rows:
        head = row[:5].strip()
        if head.isdigit():
            seen = int(head)
        if low <= seen <= high:
            out.append(row)
    return out
