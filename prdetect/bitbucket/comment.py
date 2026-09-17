"""A pull request's findings as one general comment under it.

The findings are written as a table in a single comment in the pull request's
conversation -- never as inline comments on the code. A pull request without
findings gets the same comment saying there are none, so every reviewed pull
request shows that it was reviewed. A later run updates that same comment
instead of adding another; a comment is recognised as ours by its first line,
`HEADING`.

Only what a run published reaches Bitbucket: severity, kind, file:line and the
finding's title. Nothing from an answer key is ever written.
"""
from __future__ import annotations

from collections import Counter
from typing import Protocol, Sequence

from prdetect.severity import ORDER, severity

HEADING = "### prdetect bulguları"


class CommentClient(Protocol):
    def comments(self, workspace: str, repo: str, pull_id: int) -> list[dict]: ...

    def post_comment(self, workspace: str, repo: str, pull_id: int, raw: str) -> dict: ...

    def update_comment(self, workspace: str, repo: str, pull_id: int, comment_id: int, raw: str) -> dict: ...


def _cell(text: object) -> str:
    """One table cell: on one line, with no `|` to break the row."""
    return " ".join(str(text or "").split()).replace("|", "\\|")


def _location(finding: dict) -> str:
    line = int(finding["line"])
    end = int(finding.get("end_line") or line)
    return f"{finding['file']}:{line}" if end <= line else f"{finding['file']}:{line}-{end}"


def ordered(findings: Sequence[dict]) -> list[dict]:
    """Most severe first, then by file and line."""
    return sorted(findings, key=lambda f: (ORDER.index(severity(f["type"])), f["file"], int(f["line"])))


NO_FINDINGS = "**Bulgu yok.** Otomatik inceleme bu pull request'te raporlanacak bir kusur bulmadı."
NO_SOURCE = "Not: bu pull request Python kaynağı değiştirmiyor; model yalnız Python dosyalarını inceliyor."


def render(findings: Sequence[dict], head_commit: str, model: str | None, run: str,
           reviews_source: bool = True) -> str:
    """The comment text, in Bitbucket markdown.

    With `reviews_source` False the pull request changes no file the model reads,
    and the comment says so, so that "no findings" is not taken for a clean review.
    """
    rows = ordered(findings)
    if rows:
        counts = Counter(severity(f["type"]) for f in rows)
        summary = ", ".join(f"{counts[level]} {level}" for level in ORDER if counts[level])
        body = [f"{len(rows)} bulgu ({summary}).", "",
                "| # | Ciddiyet | Tür | Konum | Açıklama |",
                "|---|---|---|---|---|"]
        body += [f"| {index} | {severity(f['type'])} | `{_cell(f['type'])}` | `{_cell(_location(f))}` "
                 f"| {_cell(f.get('message') or f.get('title'))} |"
                 for index, f in enumerate(rows, start=1)]
    else:
        body = [NO_FINDINGS] if reviews_source else [NO_FINDINGS, "", NO_SOURCE]
    footer = (f"_Otomatik inceleme: model {model or '-'}, koşu {run}. Satırlar {head_commit[:12]} "
              f"commit'ine göre; bulgular yanlış olabilir._")
    return "\n".join([HEADING, "", *body, "", footer])


def is_ours(comment: dict) -> bool:
    return (not comment.get("deleted") and not comment.get("inline")
            and (comment.get("content") or {}).get("raw", "").startswith(HEADING))


def upsert(client: CommentClient, workspace: str, repo: str, pull_id: int, raw: str) -> tuple[str, int]:
    """Create our comment, or update the one already there. Returns (action, comment id)."""
    existing = next((c for c in client.comments(workspace, repo, pull_id) if is_ours(c)), None)
    if existing is not None:
        if existing["content"]["raw"].strip() == raw.strip():
            return "unchanged", existing["id"]
        client.update_comment(workspace, repo, pull_id, existing["id"], raw)
        return "updated", existing["id"]
    return "created", client.post_comment(workspace, repo, pull_id, raw)["id"]
