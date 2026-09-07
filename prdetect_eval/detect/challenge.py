"""Stage [6]: hand each claim back with the lines it is about.

The detector's false alarms are not evenly hard. Reading the twelve reports one
v3 run made on pull requests the corpus calls clean, five were contradicted by
text already in the prompt: a claim that a result was discarded when the very
next line passed ``out=``, a claim that braces were missing from a string that
contains them, a claim about ``nan_to_num`` that its documented behaviour
denies. The detector had the evidence and asserted past it.

So this stage does not re-review the pull request. It asks one question about
one claim, with one excerpt: is this refuted by these lines? That is a far
easier judgement than finding the defect, and it is the judgement the detector
skipped.

Two things keep it from becoming a delete-everything filter:

* the verdict defaults to ``stands`` -- refuting requires quoting the text that
  contradicts the claim, and a refutation with no quote is not honoured;
* every verdict is stored, so a survivor and a casualty are both auditable and
  the stage's cost is measured in true positives lost, not assumed to be zero.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from schema import Case

from . import pack as pack_module

SYSTEM = """\
You are checking one claim about one piece of code.

A reviewer has claimed a defect at a specific line. You are given that claim and \
the lines around it. Decide one thing only: **do the lines shown contradict the \
claim?**

You are not asked whether the code is good, whether the claim is the most \
important thing to say, or whether you would have made it. Only whether what is \
printed refutes it.

Refute the claim when the excerpt plainly says otherwise: the argument the claim \
says is missing is on the next line, the character the claim says is absent is \
in the string, the function the claim describes does something else by \
definition. Quote the exact text that contradicts it.

Let the claim stand when the excerpt does not settle it. Most claims are about \
code whose callers, callees and remaining file are not shown; not being able to \
confirm a claim is not the same as refuting it, and "stands" is the answer \
whenever the excerpt is merely insufficient.

Reply with JSON only:

{"verdict": "stands", "quote": "", "reason": "..."}

- `verdict` is `stands` or `refuted`.
- `quote` is the text from the excerpt that contradicts the claim, copied \
exactly. Required to refute; leave it empty when the claim stands.
- `reason` is one short clause, under fifteen words.
"""

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["stands", "refuted"]},
        "quote": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "quote", "reason"],
    "additionalProperties": False,
}


def excerpt(case: Case, filename: str, line: int, radius: int = 12) -> list[str]:
    """The lines around a claim, numbered exactly as the detector saw them."""
    if case.head_files:
        source = case.head_files.get(filename)
        if source is None:
            return []
        added = case.added_lines.get(filename, frozenset())
        rows, _ = pack_module._numbered(source, added)
        low = max(0, line - radius - 1)
        return rows[low:line + radius]
    out: list[str] = []
    for hunk in pack_module.parse_diff(case.diff):
        if hunk.filename == filename and hunk.start - radius <= line <= hunk.end + radius:
            out += hunk.lines
    return out


def build(claim: dict, rows: Sequence[str]) -> str:
    """The user message: the claim, then the lines, and nothing else."""
    return "\n".join([
        "# Claim",
        "",
        f"In `{claim['file']}`, at line {claim['line']}:",
        "",
        f"> {claim.get('title') or claim.get('message', '')}",
        "",
        f"Kind claimed: {claim['type']}",
        "",
        "# The lines",
        "",
        "```",
        *(rows or ["(the excerpt is empty; the claim cannot be refuted from it)"]),
        "```",
        "",
    ])


def parse(text: str) -> tuple[str, str, str]:
    """``(verdict, quote, reason)``. Anything unreadable leaves the claim standing.

    A stage that silently drops a finding when the server hiccups would report a
    precision gain it did not earn, so every failure resolves toward keeping the
    detector's own answer.
    """
    try:
        document = __import__("json").loads(text)
    except Exception:
        return "stands", "", "unparseable verdict"
    if not isinstance(document, dict):
        return "stands", "", "verdict was not an object"
    verdict = str(document.get("verdict", "stands")).strip().lower()
    quote = str(document.get("quote", "") or "")
    reason = str(document.get("reason", "") or "")
    if verdict != "refuted":
        return "stands", quote, reason
    return "refuted", quote, reason


GUTTER = re.compile(r"^\s*\d*\s*[+-]?\s*\| ", re.M)


def _normalise(text: str) -> str:
    """Compare on the code, not on how either side happened to print it.

    Two mismatches were measured and both rejected a correct refutation: a quote
    copied with the line-number gutter still attached, and a backslash the model
    unescaped on its way through JSON.
    """
    text = GUTTER.sub("", text).replace("\\\\", "\\")
    return re.sub(r"\s+", " ", text).strip().lower()


def honours(quote: str, rows: Sequence[str]) -> bool:
    """Whether a refutation actually points at text in the excerpt.

    Without this the stage can refute anything by asserting a contradiction that
    is not there -- which is the same failure it exists to catch, one call later.
    """
    needle = _normalise(quote)
    if len(needle) < 3:
        return False
    haystack = _normalise(" ".join(row.split("| ", 1)[-1] for row in rows))
    return needle in haystack
