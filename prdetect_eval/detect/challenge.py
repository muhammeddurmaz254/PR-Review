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

Reply with JSON only, and **in this order**:

{"reason": "...", "quote": "", "verdict": "supports"}

- `reason` comes first, and it is where you work: say what the lines actually \
do, in one short clause.
- `quote` is text copied exactly from the excerpt. Required when the verdict is \
`contradicts`; otherwise leave it empty.
- `verdict` comes last and names what your own reason just said about the \
excerpt:
  - `supports` -- the lines show what the claim describes. Use this whenever \
your reason confirms the claim, even partly.
  - `contradicts` -- the lines show the opposite of the claim: the argument it \
says is missing is there, the call it names does something else.
  - `does_not_settle` -- the lines neither show it nor rule it out.

Only `contradicts` removes the claim. If your reason ends by agreeing with the \
claim, the verdict is `supports`, never `contradicts`.
"""

# Two things here were measured, not designed.
#
# Field order: constrained decoding emits properties in schema order, and with
# thinking disabled that order is the only room the model has. With `verdict`
# first it voted then explained; reason first produced visibly better readings.
#
# The values: `stands`/`refuted` was worse than useless. Reason-first raised the
# refutations from six to twenty and nine of those twenty carried a reason that
# *agreed* with the claim -- "explicitly assigns the string literal ... which
# constitutes a hardcoded credential", verdict `refuted`. The model read well and
# inverted the vote, because "refuted" reads as "I have something to say about
# this line". So the enum no longer contains a negation: each value names what
# the excerpt does, and only one of them removes a finding.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "quote": {"type": "string"},
        "verdict": {"type": "string",
                    "enum": ["supports", "contradicts", "does_not_settle"]},
    },
    "required": ["reason", "quote", "verdict"],
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


def build(claim: dict, rows: Sequence[str], facts: Sequence[str] = ()) -> str:
    """The user message: the claim, the lines, and what was counted.

    The facts are here because of one measured failure. `conf-03` claims a
    timeout is duplicated from a value in the settings module; the challenger saw
    `client.py` alone, could not see the other definition, concluded the constant
    was merely hardcoded and refuted a true positive. Its own instruction says a
    claim stands when the excerpt cannot settle it, and it is not reliably held
    to that -- the same reason `settleable` exists for cross-file kinds. A claim
    about two places is settled by knowing what is in both, and the counting
    stage already knows.
    """
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
        *(["# What was counted elsewhere", "",
           "Measured over the whole repository, not read from the excerpt above.",
           "", *facts, ""] if facts else []),
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
    verdict = str(document.get("verdict", "supports")).strip().lower()
    quote = str(document.get("quote", "") or "")
    reason = str(document.get("reason", "") or "")
    # Only an explicit contradiction removes a finding; every other answer,
    # including one this parser does not recognise, keeps it.
    if verdict != "contradicts":
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


def settleable(claim_type: str) -> bool:
    """Whether a one-file excerpt could contradict a claim of this kind at all.

    The stage's own instruction is that "stands" is the answer whenever the
    excerpt is merely insufficient. For a type whose name is `crossfile_*` the
    excerpt is insufficient by construction: the claim asserts a mismatch between
    two files and the excerpt shows one, so no reading of it can settle the
    question. Enforcing that mechanically is the same move as `honours` -- the
    prompt already says it, and the model is not reliably held to it.

    Measured: `data-03-kusurlu` reports that `from_minor(...)` hands major units
    to a callee documented, in the *other* file of the same pull request, to take
    minor units. The challenger saw only `billing/services.py`, read the
    conversion as deliberate, and refuted a true positive on a reason that ended
    "a callee that likely expects major units". Across the five stored halka
    challenge runs this gate spares six findings, all of them true positives, and
    costs two false alarms. It is inert on the other two corpora, whose
    taxonomies name no cross-file type.

    The better fix is to widen the excerpt so the challenger can actually answer;
    that costs a GPU run, and this gate is what the stored artefacts support.
    """
    return not claim_type.startswith("crossfile_")
