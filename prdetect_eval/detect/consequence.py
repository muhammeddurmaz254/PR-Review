"""Stage [6b]: ask what breaks, because stage [6a] has stopped catching anything.

Stage [6a] asks one question -- *do these lines contradict the claim?* -- and it
answers it well. On the broad-taxonomy ``halka`` run it removed three false
alarms and one true positive out of sixty claims, and then stopped: of the
thirteen false alarms that survived, twelve were **claimed** ``stands`` by the
challenger itself, with a quote found in the excerpt and the excerpt judged able
to settle the kind. It was not blind and it was not overruled. It agreed.

Reading its own reasons says why, and the finding is not about prompt wording:

    "404 response carries a message string"  -> "line 69 explicitly passes the
    string 'bulunamadi' ... This confirms the claim."

    "sequence field set to invoice id"  -> "line 48 explicitly sets the
    `sequence` field to `invoice.id`, which matches the claim."

    "per-row save instead of bulk update"  -> "iterates over a queryset and
    calls `invoice.save()` inside the loop, which confirms the claim."

Every one of those is **true**. Every one is on a pull request the corpus calls
clean. The detector's surviving failure mode is not fabrication -- it is an
accurate observation asserted as a defect. A node built to catch a claim the
text denies cannot catch a claim the text confirms, and no rewording of
"contradict" will make it, because the answer to that question is genuinely
*yes, the lines say that*.

So the refutation node is not revised; it is split. This is the second half, and
it asks the question the first half structurally cannot:

    Name the run in which this goes wrong.

Not whether the code is ideal, not whether the observation is accurate -- what
concretely happens: which caller, with what value, reaching which line, and what
the program then does that it should not. A real defect answers it in one
sentence. A true remark about correct code has to invent a caller, and the
invention is visible.

The same two protections as [6a], in the same directions:

* **The default keeps the finding.** Only an answer that names the harm as
  ``style_only`` or ``none`` removes one; an unparseable answer, a server error
  or an enum this parser does not recognise all leave the detector's own report
  standing.
* **A verdict in either direction must point at something.** ``honours`` refuses
  a refutation that quotes nothing; ``concrete`` refuses a *survival* that
  triggers nothing -- a claim whose harm is asserted but whose trigger is blank
  or literally "none" is an assertion about a run the model could not describe.
  That reading is strictly harsher, so it is stored rather than applied, and
  ``run_regate.py`` can score both from one pass.
"""
from __future__ import annotations

from typing import Any, Sequence

SYSTEM = """\
A reviewer has left one comment on one line of a pull request. You are given the \
comment and the lines around it.

Answer one question: **name the run in which this goes wrong.**

Not whether the comment is accurate -- assume it is. Not whether the code could \
be written better. One concrete execution: what drives the program to this line, \
and what it then does that it should not.

The thing that drives it is an input to the *system*: a request from a \
particular user, a queued job, a row already in the database, a retry, two \
requests at once. **It is almost never visible in the excerpt, and that is \
normal.** You are shown a few lines of one file; the caller lives elsewhere. \
Being unable to see the caller is not a reason to say there is no trigger. Name \
the ordinary way this code is reached.

Two comments of the same shape, one of each answer:

    "endpoint checks the read permission" -- trigger: a member of the \
organisation sends the request for an invoice they may read but not void; \
consequence: the invoice is voided by someone with no right to void it. \
harm: unsafe_access.

    "404 response carries a message string" -- trigger: any request for a \
missing invoice; consequence: the caller receives "bulunamadi" and a 404, which \
is what a 404 is for. harm: none.

The difference is not how visible the caller is. It is that the first has a run \
whose result is wrong and the second does not.

Say `none` when the line behaves correctly for every input you can think of -- \
not when you cannot see who calls it.

Reply with JSON only, and **in this order**:

{"trigger": "...", "consequence": "...", "harm": "wrong_behaviour"}

- `trigger` comes first: the request, job, stored row or timing that reaches this \
line and makes it matter. Write the ordinary one. Write `none` only if every \
input reaches it harmlessly.
- `consequence`: what the program then does that it should not -- the wrong \
value, the access that should have been refused, the work that never finishes. \
Write `none` if the program behaves correctly.
- `harm` comes last and names what your own two answers just described:
  - `wrong_behaviour` -- the program computes, stores or returns something wrong.
  - `unsafe_access` -- someone reaches data or an action that should be refused.
  - `resource_cost` -- it completes correctly but at a cost that will not hold.
  - `style_only` -- the observation is true and the program is correct: it reads \
worse, duplicates something, or is named badly.
  - `none` -- there is no input for which this misbehaves.

All five are ordinary answers. Choose the one your own `trigger` and \
`consequence` describe, whichever it is.
"""

# Field order is generation order under constrained decoding with thinking off,
# and it is the whole design here: `harm` is last so the label names a run the
# model has already had to write down, rather than a verdict it then justifies.
# This is the same ordering lesson VERDICT_SCHEMA records, applied to a question
# whose answer is harder to fake -- a claim needs a caller and a value, and both
# are printed before the vote.
HARM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trigger": {"type": "string"},
        "consequence": {"type": "string"},
        "harm": {"type": "string",
                 "enum": ["wrong_behaviour", "unsafe_access", "resource_cost",
                          "style_only", "none"]},
    },
    "required": ["trigger", "consequence", "harm"],
    "additionalProperties": False,
}

#: The two answers that remove a finding. Everything else keeps it, including an
#: answer this module does not recognise.
HARMLESS = frozenset({"style_only", "none"})

#: What a model writes when it means "nothing". Kept short and literal: a phrase
#: test would start deleting findings for hedging, which is not the same thing.
EMPTY = frozenset({"", "none", "n/a", "na", "null", "nothing", "-", "hicbiri", "yok"})


def build(claim: dict, rows: Sequence[str], facts: Sequence[str] = ()) -> str:
    """The user message: the comment, the lines, and what was counted elsewhere.

    Deliberately the same payload as stage [6a]. The two nodes differ in the
    question, not in the evidence, so a difference in their answers is a
    difference in what was asked.
    """
    return "\n".join([
        "# The comment",
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
        *(rows or ["(the excerpt is empty)"]),
        "```",
        "",
        *(["# What was counted elsewhere", "",
           "Measured over the whole repository, not read from the excerpt above.",
           "", *facts, ""] if facts else []),
    ])


def parse(text: str) -> tuple[str, str, str]:
    """``(harm, trigger, consequence)``. Anything unreadable keeps the finding.

    A stage that silently drops a report when the server hiccups would report a
    precision gain it did not earn, so every failure resolves toward the
    detector's own answer -- the same rule as ``challenge.parse``.
    """
    try:
        document = __import__("json").loads(text)
    except Exception:
        return "unparseable", "", ""
    if not isinstance(document, dict):
        return "unparseable", "", ""
    harm = str(document.get("harm", "") or "").strip().lower()
    trigger = str(document.get("trigger", "") or "").strip()
    consequence = str(document.get("consequence", "") or "").strip()
    return harm, trigger, consequence


def harmless(harm: str) -> bool:
    """Whether this answer removes the finding. Unknown values never do."""
    return harm in HARMLESS


def concrete(trigger: str, consequence: str) -> bool:
    """Whether an answer claiming harm actually described a run.

    The mirror of ``challenge.honours``. That gate refuses a refutation that
    quotes nothing, because a removal has to point at text; this refuses a
    survival that triggers nothing, because a claim of harm has to point at a
    run. It is the harsher of the two readings and it is measured, not assumed:
    the runner stores it and applies it only when asked.
    """
    return _said_something(trigger) and _said_something(consequence)


def _said_something(text: str) -> bool:
    stripped = text.strip().rstrip(".").strip().lower()
    return stripped not in EMPTY and len(stripped) > 3
