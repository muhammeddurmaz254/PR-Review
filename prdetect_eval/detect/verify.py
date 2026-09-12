"""The verifier agent: one claim, a fresh context, tools, and a narrow question.

Replaces the challenge stage. The challenger sees a window of lines around a
claim and nothing else, so it can only ask whether that window contradicts
the claim; a claim that is true on its surface and wrong about the code it
depends on passes. The open reviewer showed that tools find what a window
cannot -- perf-03's batch selector sits in an unchanged file, and a search
for it produced the labelled finding -- and that left open, the same tools
produce new suspicions instead of evidence. So here the tools are bound to
one claim and a budget of four calls, and the context holds nothing but that
claim.

The verdict is last in the schema and its values carry no negation: with
constrained decoding the schema order is the generation order, and both
facts were measured on the challenge stage (a verdict before its reason, or a
negated enum value, flipped votes). What the verifier cites is checked
mechanically afterwards -- that is layer 3 -- so a verdict it cannot back with
lines that exist is treated as unsettled.
"""
from __future__ import annotations

MEASURED = """\
Against the challenge stage it replaces, on the same claims:

    halka   44/9/4  ->  45/5/3 strict (F1 0.871 -> 0.918), 45/6/3 lenient
    zincir   8/3/8  ->   5/2/11 strict (F1 0.593 -> 0.435),  7/3/9 lenient

halka is the best result this pipeline has had: all forty-five true claims
were established, and five of its ten false ones were removed. zincir lost
four true claims, and each says the same thing in its reason -- no code was
found that reads the two halves together. `replay-01-k1`: the key does carry
`attempt`, "but I could not find code that uses key_for as a dedup guard".
`bus-01-k1` and `profile-01-k1`: "no consumer couples them". That is the
standard asked for, applied to defects whose coupling this repository states
in a docstring rather than executes. `dlq-01-k1` is different and fair: the
key is not removed, the missing thing is the table entry it now reads from,
and the claim said otherwise.

Layer 3 changed no verdict on either corpus: every decisive answer cited
lines that are really there. What the verifier gets wrong, it gets wrong with
the evidence in front of it.
"""

import json
from typing import Sequence

from . import anchor

VERDICTS = ("established", "contradicted", "unsettled")

SYSTEM = """\
You check one claim someone made about a pull request: that it introduces a particular defect at a particular line. You did not make the claim, and you have no stake in it.

Decide it from code you have actually seen. You are shown the lines around the claim, and you can read the repository at this pull request with tools: read a file, read it as it was before the change, find where a name is defined or used, search for text. Look only for what this one claim needs.

Answer in this order:
- `needed`: what would have to be true in the code for the claim to hold -- the thing you need to see.
- `evidence`: the lines you rely on, each as `file`, `line` and `quote` (the code copied exactly, without the line number or mark). Only lines you have seen.
- `reason`: one or two sentences connecting the evidence to your verdict.
- `verdict`: `established` if the lines you cite show the defect; `contradicted` if the lines you cite show the claim is wrong; `unsettled` if you could not find lines that decide it either way.

A claim you cannot back with lines is `unsettled`, however plausible it sounds.
"""

GUIDE = """
You have up to four tool calls. Use them on what `needed` names, then stop; you will be asked for your answer.
"""

FINAL_ASK = "Answer now, as JSON with `needed`, `evidence`, `reason` and `verdict`, in that order."

SCHEMA = {
    "type": "object",
    "properties": {
        "needed": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "file": {"type": "string"}, "line": {"type": "integer"}, "quote": {"type": "string"}},
            "required": ["file", "line", "quote"]}},
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["needed", "evidence", "reason", "verdict"],
}


def user_message(claim: dict, definition: str, rows: Sequence[str]) -> str:
    return "\n".join([
        "# The claim", "",
        f"In `{claim['file']}`, at line {claim['line']}: {claim.get('message') or claim.get('title', '')}", "",
        f"Kind claimed: `{claim['type']}` -- {definition}", "",
        "# The lines around it", "", "```", *(rows or ["(no lines available)"]), "```", "",
        "Decide whether this pull request really introduces this defect. First name what you need "
        "to see; look for it; then answer.",
    ])


def parse(text: str) -> dict:
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"verdict": "unsettled", "evidence": [], "reason": "unreadable answer", "needed": ""}
    if document.get("verdict") not in VERDICTS:
        document["verdict"] = "unsettled"
    document["evidence"] = [e for e in document.get("evidence") or [] if isinstance(e, dict)]
    return document


def cited(workspace, item: dict, slack: int = 3) -> bool:
    """Layer 3: the cited code is really at (or next to) the cited line.

    Checked against the file after the change and, failing that, before it --
    a verifier may rightly cite a line the change deleted.
    """
    quote = anchor.normalise(str(item.get("quote", "")))
    if len(quote) < 4:
        return False
    path = str(item.get("file", "")).strip().removeprefix("./")
    try:
        line = int(item.get("line", 0))
    except (TypeError, ValueError):
        return False
    for files in (workspace.head, workspace.base):
        rows = files.get(path, "").split("\n")
        window = rows[max(0, line - 1 - slack): line + slack]
        if any(quote in anchor.normalise(row) or (anchor.normalise(row) and anchor.normalise(row) in quote)
               for row in window):
            return True
    return False


def settle(answer: dict, workspace) -> str:
    """The verdict layer 3 lets stand: a decisive verdict needs evidence that exists."""
    verdict = answer.get("verdict", "unsettled")
    if verdict == "unsettled":
        return verdict
    return verdict if any(cited(workspace, e) for e in answer.get("evidence") or []) else "unsettled"


def keeps(verdict: str, policy: str) -> bool:
    """strict: only what was established is published. lenient: only what was contradicted is dropped."""
    return verdict == "established" if policy == "strict" else verdict != "contradicted"
