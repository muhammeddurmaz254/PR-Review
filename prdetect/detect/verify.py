"""The verifier: one claim, a fresh context, tools, and a narrow question.

Every claim the detector makes is checked on its own. The verifier is shown the
claim and the lines around it, and gets read-only tools over the repository
(`agent.py`) with a budget of calls. It names what it needs to see, cites the
lines it relied on, and answers `established`, `contradicted` or `unsettled`. The
verdict comes last in the answer schema: constrained decoding generates in schema
order, and a reason written after its verdict can only defend it.

What it cites is then checked mechanically: a decisive verdict whose quotes are
not at the lines it names is treated as `unsettled`.

A claim can be put two ways. `user_message` gives the catalogue name and its
definition; `user_message_located` gives only the detector's own sentence, so a
name that fits badly cannot decide the verdict. The pipeline asks both and
publishes what both establish (`agreed`).
"""
from __future__ import annotations

import json
from typing import Sequence

from prdetect.detect import anchor

VERDICTS = ("established", "contradicted", "unsettled")

SYSTEM = """\
You check one claim someone made about a pull request: that it introduces a particular defect at a particular line. You did not make the claim, and you have no stake in it.

Decide it from code you have actually seen.
A rule this repository states in prose counts as evidence like any line of code: a docstring, a comment or a documented contract saying what a value or a function has to do. A change that breaks such a rule is a defect even when nothing else in the repository enforces it. Quote the prose the way you quote code.
 You are shown the lines around the claim, and you can read the repository at this pull request with tools: read a file, read it as it was before the change, find where a name is defined or used, search for text. Look only for what this one claim needs.

Answer in this order:
- `needed`: what would have to be true in the code for the claim to hold -- the thing you need to see.
- `evidence`: the lines you rely on, each as `file`, `line` and `quote` (the code copied exactly, without the line number or mark). Only lines you have seen.
- `reason`: one or two sentences connecting the evidence to your verdict.
- `verdict`: `established` if the lines you cite show the defect; `contradicted` if the lines you cite show the claim is wrong; `unsettled` if you could not find lines that decide it either way.

Before you decide, write the failure out: the input or the order of events that produces it, and the line where the wrong thing happens. Not what could go wrong in general -- what goes wrong here: a value a caller can actually supply, two requests that can actually overlap, a branch that can actually be taken. If the code you have seen does not let you write that run -- the value cannot reach that line, the branch cannot be taken, the state cannot be built -- the verdict is `contradicted`. A defect no run can reach is not a defect. If the code still lets it happen, the claim stands even where the rest of the repository does the same thing.

Whether anything in this repository calls the code is not the question, and it never contradicts a claim on its own. A function a change adds is reached by a route, a schedule, a handler, a test, a command, or by the change that comes after this one; the repository you can see is not the whole of it. Ask only whether the wrong thing happens when the line runs with a value a caller could pass -- not whether you can find that caller.

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
    """The claim with its catalogue name and that name's definition."""
    return "\n".join([
        "# The claim", "",
        f"In `{claim['file']}`, at line {claim['line']}: {claim.get('message') or claim.get('title', '')}", "",
        f"Kind claimed: `{claim['type']}` -- {definition}", "",
        "# The lines around it", "", "```", *(rows or ["(no lines available)"]), "```", "",
        "Decide whether this pull request really introduces this defect. First name what you need "
        "to see; look for it; then answer.",
    ])


def user_message_located(claim: dict, definition: str, rows: Sequence[str]) -> str:
    """The claim without its catalogue name: the sentence is the claim.

    A claim with no sentence of its own falls back to the named form -- there is
    nothing else to judge it by.
    """
    sentence = (claim.get("message") or claim.get("title") or "").strip()
    if not sentence:
        return user_message(claim, definition, rows)
    return "\n".join([
        "# The claim", "",
        f"In `{claim['file']}`, at line {claim['line']}: {sentence}", "",
        "# The lines around it", "", "```", *(rows or ["(no lines available)"]), "```", "",
        "Decide whether this pull request really introduces the problem that sentence describes, at "
        "this place. The sentence is the whole claim: judge whether the code shows it, nothing wider "
        "and nothing narrower. First name what you need to see; look for it; then answer.",
    ])


def agreed(claims: Sequence[dict], *verdicts: dict) -> list[dict]:
    """Claims every verifier established, in their original order.

    Each of `verdicts` maps `(case_id, file, line, type)` to a verdict row. A
    claim one verifier never saw is not agreed on.
    """
    key = lambda c: (c["case_id"], c["file"], c["line"], c["type"])
    return [c for c in claims
            if all((v.get(key(c)) or {}).get("verdict") == "established" for v in verdicts)]


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
    """The cited code is really at (or next to) the cited line.

    Checked against the file after the change and, failing that, before it: a
    verifier may rightly cite a line the change deleted.
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
    """The verdict that stands: a decisive verdict needs evidence that exists."""
    verdict = answer.get("verdict", "unsettled")
    if verdict == "unsettled":
        return verdict
    return verdict if any(cited(workspace, e) for e in answer.get("evidence") or []) else "unsettled"
