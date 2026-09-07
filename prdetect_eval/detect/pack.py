"""Turn one case into the text the model reads. One call per pull request.

There is no candidate enumeration stage. It was measured away: a demo_repo case
changes 1.4 files averaging 356 tokens of content, and a SWRBench case is a
single hunk of 88. Selecting sites inside that would add a stage that can only
lose findings -- and eight of demo_repo's eighteen defects exist only in the
relationship between two changed files, which per-file packing would make
unreachable by construction.

Two shapes are handled. A case with file contents is printed whole, with real
line numbers and a `+` on the lines the pull request touched. A case without
them -- SWRBench keeps only diff snippets, since the alternative is cloning
eleven upstream repositories -- is shown as the snippet it has.
"""
from __future__ import annotations

from dataclasses import dataclass

from schema import Case

from . import prompt as prompt_module


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


def build(case: Case, dataset: str) -> Pack:
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
        # No checkout available: the diff snippet is the whole of the evidence.
        body += ["# CHANGED CODE", "",
                 "Only the changed hunks are available for this pull request. "
                 "Line numbers are the file's own.", "",
                 "```", case.diff.strip(), "```", ""]
        shown = case.diff.count("\n") + 1

    return Pack(
        case_id=case.case_id,
        system=prompt_module.system(dataset),
        user="\n".join(body),
        shown_lines=shown,
    )
