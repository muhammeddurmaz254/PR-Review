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
Against the challenge stage it replaces, on the same claims, strict policy
(only what was established is published), with the contract clause:

    halka       44/9/4  ->  45/5/3   F1 0.871 -> 0.918   precision 0.830 -> 0.900
    demo_repo   16/4/1  ->  16/3/1   F1 0.865 -> 0.889   precision 0.800 -> 0.842
    zincir       8/3/8  ->   6/1/10  F1 0.593 -> 0.522   precision 0.727 -> 0.857

Precision rises on all three corpora. halka and demo_repo keep every true
claim they had -- halka gains one and loses four false alarms, including ones
every prompt version and both model families agreed on. zincir pays two true
claims for two false ones.

Its losses are worth reading, because they are the standard working. On
`dlq-01-k1` the verifier is right and the claim was wrong: the key is still
there, what is missing is the table entry it now reads from. On `profile-01`
it disagrees with the corpus on the merits -- a lower backoff cap in the
production profile is "a legitimate per-environment choice" -- and it says so
with the lines in front of it. `bus-01` it could not settle inside four tool
calls.

Without the contract clause zincir was 5/2/11 and halka unchanged at 45/5/3:
the clause recovered `replay-01-k1`, whose contract is stated in a docstring
("a replayed event is THE SAME event"), and cost nothing on halka.

Layer 3 changed no verdict on any corpus: every decisive answer cited lines
that are really there. What the verifier gets wrong, it gets wrong with the
evidence in front of it.
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

# A rule the repository writes down is a rule. Every zincir claim the verifier
# lost gave the same reason -- no code reads the two halves together -- on
# defects whose coupling this repository states in a docstring: "a replayed
# event is THE SAME event, the key must be the same". Prose is not weaker
# evidence than a consumer; it is the statement of intent a consumer would
# only imply. Kept as a separate text so the measured one above is untouched,
# and chosen with `run_verify.py --contract-evidence`.
CONTRACT_CLAUSE = """
A rule this repository states in prose counts as evidence like any line of code: a docstring, a comment or a documented contract saying what a value or a function has to do. A change that breaks such a rule is a defect even when nothing else in the repository enforces it. Quote the prose the way you quote code.
"""

SYSTEM_CONTRACT = SYSTEM.replace(
    "Decide it from code you have actually seen.",
    "Decide it from code you have actually seen." + CONTRACT_CLAUSE, 1)

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
