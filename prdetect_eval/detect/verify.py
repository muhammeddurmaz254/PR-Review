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

# D17 asked what the catalogue's precedent clauses are for. Thirty of the
# seventy-four definitions only name a defect where the repository already
# shows the right way -- "the other call sites", "sibling mutations of the same
# state" -- and taking those words out to make the catalogue general (v11, v12)
# raised halka's false alarms from four to nine while recovering demo_repo's
# unnameable findings. So the clause does two jobs at once: it names the defect
# and it suppresses the claim. Naming has to happen in the catalogue. Checking
# the precedent does not: this stage has the tools to go and look, and it can
# read the revision before the change, which a detector reading one pack never
# sees -- a guard this pull request deleted is a precedent the catalogue could
# not have known about.
PRECEDENT_CLAUSE = """
A claim that something is missing -- a lock, a check, a guard, a timeout, a validation -- is settled by how this repository does the same thing elsewhere. Go and find the comparable place: another mutation of the same state that takes the lock, another call site that checks first, the same operation as it stood in the revision before this change. If you find it, the claim is established. If the comparable places do the same thing as the changed code, and nothing in the repository states the rule, the verdict is `contradicted`: what is missing there is missing everywhere, and this pull request did not introduce it.
"""

# MEASURED (D18). Family rung, same claims in, only this stage's prompt differs:
#
#     on v12 claims   halka 44/9/4 -> 44/8/4    zincir 7/2/9 -> 7/1/9   demo 12/3/5 unchanged
#     on v10 claims   halka 44/4/4 -> 45/9/3    zincir 7/1/9 -> 8/1/8   demo  9/2/8 unchanged
#
# It moves recall up and precision down, and which way that lands depends on
# the catalogue it runs behind: pooled F1 0.797 -> 0.808 on v12, 0.811 -> 0.800
# on v10. It is not an independent improvement and is off by default.
#
# It does NOT pay for taking the precedent out of the catalogue. With every
# definition made intrinsic (review/v13-universal) halka's false alarms went
# 8 -> 14 with this clause on. Reading them says why: asked to find the
# comparable place, the verifier finds one and uses it to CONFIRM. On
# `authz-02-temiz` it reported that the new endpoint checks a list-level
# permission "whereas the comparable single-record endpoint invoice_detail
# checks the per-record permission" -- a precedent argument for the defect, on
# a case the corpus calls clean. Four tool calls will find a comparable site
# for almost any claim; what the catalogue's clause did was stop the claim
# from being made at all. Suppression has to happen where the claim is made.
SYSTEM_PRECEDENT = SYSTEM_CONTRACT.replace(
    "A claim you cannot back with lines is `unsettled`, however plausible it sounds.",
    PRECEDENT_CLAUSE.strip()
    + "\n\nA claim you cannot back with lines is `unsettled`, however plausible it sounds.", 1)


# The precedent clause asks how the rest of the repository does the same thing,
# and D18 measured what that is worth. It also has a defect no measurement on
# our corpora shows: it defines a bug as a departure from the repository's own
# standard, so a repository with no standard has no bugs. A service that locks
# nothing anywhere still loses money to a double spend. Nothing that ships can
# ask that question.
#
# This asks the opposite kind of question, and asks nothing of the repository:
# write the failure out. A claim whose failing run cannot be written from the
# code in front of you is not a finding, whatever its shape suggests -- and a
# claim whose failing run CAN be written stands even if every file in the
# repository has the same hole.
MECHANISM_CLAUSE = """
Before you decide, write the failure out: the input or the order of events that produces it, and the line where the wrong thing happens. Not what could go wrong in general -- what goes wrong here: a value a caller can actually supply, two requests that can actually overlap, a branch that can actually be taken. If the code you have seen does not let you write that run -- the value cannot reach that line, the branch cannot be taken, the state cannot be built -- the verdict is `contradicted`. A defect no run can reach is not a defect. If the code still lets it happen, the claim stands even where the rest of the repository does the same thing.
"""

# MEASURED (D19), on review/v13-universal's claims, family rung, against the
# same claims through the plain verifier:
#
#     halka      43/14/5 -> 41/11/7     zincir 8/2/8 -> 7/2/9
#     demo_repo  12/4/5  -> 13/3/4      pooled F1 0.768 -> 0.772, precision 0.759 -> 0.792
#
# It removes three of halka's false alarms and two of its true findings. The
# same claims through the precedent clause gave halka 43/14/5 -- identical to
# the plain verifier, which is the measurement that matters here: asking the
# verifier a different question about the repository changed nothing, and
# asking it to write the failing run changed a little. The intrinsic
# catalogue's false alarms are made at detect time and are not removable by
# rewording this stage.
SYSTEM_MECHANISM = SYSTEM_CONTRACT.replace(
    "A claim you cannot back with lines is `unsettled`, however plausible it sounds.",
    MECHANISM_CLAUSE.strip()
    + "\n\nA claim you cannot back with lines is `unsettled`, however plausible it sounds.", 1)


# D23. The clause above cost four true findings on halka and zincir, and three
# of the four fell to one argument: nothing in the repository calls it.
#
#   ssrf-02: "The function does fetch a caller-supplied URL with no validation
#             and follows redirects, which matches the described pattern.
#             However, `ping_callback` is defined but never called anywhere in
#             the repository ... A defect no run can reach is not a defect."
#   bus-01:  "no code reads them together ... no such consumer exists in the
#             repository."
#
# In each the verifier states the defect and then rejects it for want of a
# caller. That is the wrong test: the caller of code a change adds is usually
# not in the change and often not in the repository at all -- a route table, a
# scheduler, a framework, a test, or the pull request that comes next. The
# reachability question is worth asking of a *value* (can this input get
# here?), never of a *definition* (does anyone call this?).
#
# This keeps the run but takes the caller argument away from it.
# MEASURED (D23), same claims, family rung, against --mechanism alone:
#
#     halka 41/8/7 -> 43/11/5     zincir 7/2/9 -> 8/2/8    demo_repo 13/2/4 -> 13/3/4
#     with the 0.8 confidence floor: pooled 60/10/21 -> 63/13/18,
#     F1 0.795 -> 0.803, recall 0.741 -> 0.778, precision 0.857 -> 0.829.
#
# The three findings it was written for came back: ssrf-01 and ssrf-02, whose
# fetches the verifier had described correctly and then dismissed for want of a
# caller, and dlq-01. It costs three false alarms, all on clean twins and all
# of the "a guard is missing here" kind the caller argument had been suppressing
# by accident.
#
# Adopted on those numbers and on the argument itself: "nothing in this
# repository calls it" is not a reason to reject a review comment, and a rule
# that is wrong does not get to stay because it happened to silence noise --
# that is what the precedent clause was dropped for. It misfires more, not
# less, outside the corpora: the caller of code a pull request adds is almost
# never in the pull request.
CALLERS_NOTE = """
Whether anything in this repository calls the code is not the question, and it never contradicts a claim on its own. A function a change adds is reached by a route, a schedule, a handler, a test, a command, or by the change that comes after this one; the repository you can see is not the whole of it. Ask only whether the wrong thing happens when the line runs with a value a caller could pass -- not whether you can find that caller.
"""

SYSTEM_MECHANISM_CALLERS = SYSTEM_MECHANISM.replace(
    "A claim you cannot back with lines is `unsettled`, however plausible it sounds.",
    CALLERS_NOTE.strip()
    + "\n\nA claim you cannot back with lines is `unsettled`, however plausible it sounds.", 1)


# D31. The verifier judged the name, not the line. On the two sets this project
# was never tuned on, it contradicted claims that stood on a labelled defect
# under the wrong kind and nothing else found them: three on stock_bench, three
# on the zincir holdout -- against none on halka and demo_repo and one on zincir
# dev. `stock-11` is the plainest: the reason describes the defect word for word
# ("none of them assert that a non-multiple amount is fully covered") and the
# verdict is `contradicted`, because the claim said `off_by_one`.
#
# On a repository the catalogue was written from, the detector's names fit; on
# one it was not, the detector finds the place and misnames what is there. So
# this asks for a second answer beside the verdict: which kind the evidence
# actually shows. It is narrow on purpose -- D26 measured that on unseen cases
# the verifier's most valuable work is removing false alarms, nine of them for
# one true finding -- and says so: no hunting for a different defect to rescue
# the claim, only a name for what the lines already cited show.
# MEASURED (D31) and NOT DEFAULT. Same claims re-verified with this clause and
# `kind`, published with the rules through fill_gaps, location rung, 0.8 floor:
#
#     stock_dev   10/3/12 -> 11/3/11   F1 0.571 -> 0.611   stock-11 recovered
#     zincir_dev  11/2/5  unchanged
#     halka       42/8/6  -> 42/7/6    F1 0.857 -> 0.866
#     demo_repo   15/1/2  -> 15/2/2    F1 0.909 -> 0.882
#     pooled      78/14/25 -> 79/14/24  F1 0.800 -> 0.806, recall 0.757 -> 0.767
#
# Five claims were renamed and four land on labels -- `stock-11` off_by_one ->
# lossy_conversion recovers a finding, halka's `inj-02` path_traversal ->
# command_injection names its label exactly. But demo_repo gains a false alarm
# that is no rename at all: a `float_money` claim on `coklu-dosya-cift-harcama`
# that the plain verifier contradicts ("the float's rounding never reaches a
# stored or charged value") is established under this clause, same kind, and
# established again on a repeat run -- so it is the clause, not run-to-run
# variance. Telling the verifier to judge the line rather than the name also
# lowers its bar on a borderline claim whose name was right. Rule 2 fails,
# reproducibly, by that one claim.
KIND_CLAUSE = """
The kind named in the claim can be wrong while the line is still defective. Judge the line, not the name. If the lines you cite show that this pull request introduces a defect at this place, the verdict is `established` even when it is a different kind from the one claimed, and `kind` is the name from the list that fits what you found. If they show no defect here -- neither the claimed one nor any other -- the verdict is `contradicted` and `kind` is `none`. Do not go looking for some other defect to rescue a claim: name a different kind only for what the lines you already cite show.
"""

FINAL_ASK_KIND = ("Answer now, as JSON with `needed`, `evidence`, `reason`, `kind` and `verdict`, "
                  "in that order.")


def schema_with_kinds(kinds: Sequence[str]) -> dict:
    """SCHEMA with `kind` before `verdict`. Constrained decoding generates in
    schema order, so the name of what was found is written before the decision
    that depends on it -- the same reason the verdict already comes last."""
    properties = {key: value for key, value in SCHEMA["properties"].items() if key != "verdict"}
    properties["kind"] = {"type": "string", "enum": [*kinds, "none"]}
    properties["verdict"] = SCHEMA["properties"]["verdict"]
    return {"type": "object", "properties": properties,
            "required": [*SCHEMA["required"][:-1], "kind", "verdict"]}


def with_kind_clause(system: str) -> str:
    return system.replace(
        "A claim you cannot back with lines is `unsettled`, however plausible it sounds.",
        KIND_CLAUSE.strip() + "\n\nA claim you cannot back with lines is `unsettled`, however plausible it sounds.", 1)


def kinds_line(kinds: Sequence[str]) -> str:
    return ("Kinds you may name, if the defect the lines show is not the one claimed: "
            + ", ".join(f"`{name}`" for name in kinds) + ".")


def published_type(claim: dict, row: dict, kinds: Sequence[str]) -> str:
    """The kind a kept claim is published under.

    Renamed only when the verifier established a defect at the line and named a
    different, valid kind for it. A contradicted or unsettled claim is never
    renamed, and `none` never replaces anything.
    """
    kind = row.get("kind")
    if row.get("verdict") == "established" and kind and kind != "none" and kind in kinds:
        return kind
    return claim["type"]


def system_for(contract: bool, precedent: bool = False, mechanism: bool = False,
               callers: bool = False) -> str:
    """The verifier's system prompt. Both added clauses imply the contract one:
    a rule the repository writes down is evidence either way. They ask opposite
    questions and are not combined -- one settles a claim by the repository's
    habits, the other by whether the failure can be reached at all."""
    if precedent and mechanism:
        raise ValueError("--precedent and --mechanism ask different questions; choose one")
    if precedent:
        return SYSTEM_PRECEDENT
    if mechanism:
        return SYSTEM_MECHANISM_CALLERS if callers else SYSTEM_MECHANISM
    return SYSTEM_CONTRACT if contract else SYSTEM


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


def user_message(claim: dict, definition: str, rows: Sequence[str],
                 kinds: Sequence[str] | None = None) -> str:
    lines = [
        "# The claim", "",
        f"In `{claim['file']}`, at line {claim['line']}: {claim.get('message') or claim.get('title', '')}", "",
        f"Kind claimed: `{claim['type']}` -- {definition}", "",
        "# The lines around it", "", "```", *(rows or ["(no lines available)"]), "```", "",
        "Decide whether this pull request really introduces this defect. First name what you need "
        "to see; look for it; then answer.",
    ]
    if kinds:
        lines += ["", kinds_line(kinds)]
    return "\n".join(lines)


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
