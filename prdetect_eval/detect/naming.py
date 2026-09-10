"""Stage [4b]: give a finding a name it was not shown.

The detector that produced these findings had no catalogue. It said what goes
wrong in its own words, which is the half of the job that does not need a
vocabulary. This stage supplies the vocabulary, and it does so as a closed
question about one finding rather than as a list held open while the code is
read.

Every measurement behind the taxonomy work points here. Putting fifty-four kinds
in the detector's prompt cost 0.242 of F1 and eleven of eighteen false alarms
arrived under kinds the corpus does not contain -- breadth in front of the
reader manufactures suspicion. A sixteen-rule conventions document did the same
thing for the same reason. Meanwhile the one stage that gains on every corpus is
the challenge, which succeeds precisely because it asks a closed question about
a claim that already exists. Naming is that shape.

The pool can therefore be as wide as a deployment likes: it is never in front of
the code. What reaches the model is the handful of candidates a lexical search
puts nearest the finding's own words, plus the option of refusing all of them --
without which the answer is a label manufactured to fill a required field.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from . import prompt as prompt_module

NONE = "none_of_these"

WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
# Words that appear in almost every definition carry no signal and would let a
# finding match on "the value is passed".
STOP = frozenset("""
the that this than them they there here into onto with without within from for
and not but its it is are was were be been being have has had does did done
one two other another same each every any all some none nothing something
code line lines file files call calls called caller callee value values
""".split())


def pool(datasets: Sequence[str] = ("halka", "demo_repo", "swrbench"),
         version: str = "review/v6") -> dict[str, str]:
    """Every kind every configured dataset names, in one catalogue.

    A deployment against an unknown repository has exactly this problem: it
    cannot know which kinds occur, so it carries all of them. Fifty-four here is
    the simulation of that, and the point of the stage is that carrying them
    costs nothing until a finding needs one.
    """
    _, taxonomies = prompt_module.VERSIONS[version]
    catalogue: dict[str, str] = {}
    for dataset in datasets:
        catalogue.update(taxonomies[dataset])
    return catalogue


# Enough morphology to match a finding's words against a definition's: "random"
# to "randomness", "fetched" to "fetch", "validation" to "validate". Measured
# without it, "Invite code generated with random.choice" scored zero against
# every kind and the ranking fell back to alphabetical order.
SUFFIXES = ("ations", "ation", "ness", "ing", "ed", "es", "s", "ly", "er", "or")


def _stem(word: str) -> str:
    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> list[str]:
    return [_stem(word) for word in WORD.findall(text.lower()) if word not in STOP]


def rank(finding: Mapping[str, Any], catalogue: Mapping[str, str],
         limit: int = 5) -> list[str]:
    """The candidate kinds nearest this finding's own words.

    Plain inverse-document-frequency overlap, no dependency and no model. The
    retrieval only has to put the right kind somewhere in a short list; choosing
    from that list is the model's job, and refusing the list is allowed, so a
    miss here is recoverable in a way a wrong choice would not be.

    Measured against the forty-three findings a real run located, queried with
    the titles that run actually wrote: the right kind is first 67% of the time,
    in the top three 86%, and in the top five 91%, where it stops -- widening to
    twelve adds nothing. Five is therefore the default and 91% is this stage's
    ceiling, six points below the 97.7% a catalogue in the prompt reaches.

    The four it cannot reach all fail the same way: the finding and the
    definition name one thing in different words -- "MD5 used for webhook
    signature verification" against "a broken digest", "Total recomputed" against
    "a second copy". Lexical overlap cannot cross that and an embedding would, so
    the ceiling is a property of this retrieval rather than of the architecture.
    """
    documents = {name: _terms(f"{name.replace('_', ' ')} {text}")
                 for name, text in catalogue.items()}
    appears = Counter(term for terms in documents.values() for term in set(terms))
    total = len(documents) or 1
    query = Counter(_terms(f"{finding.get('title', '')} {finding.get('quote', '')}"))
    if not query:
        return list(catalogue)[:limit]

    def score(terms: Sequence[str]) -> float:
        seen = set(terms)
        return sum(count * math.log(total / (1 + appears[term]))
                   for term, count in query.items() if term in seen)

    ordered = sorted(documents, key=lambda name: (-score(documents[name]), name))
    return ordered[:limit]


SYSTEM = """\
You are giving one review finding the name of its kind.

Somebody has already read the code and written down what is wrong. You are not \
re-reviewing it and you are not deciding whether it is a real defect: you are \
choosing, from a short list of kinds, the one that describes what they found.

You are given the finding's own description, the line it accuses, and the \
candidate kinds with what each one means.

## How to choose

Read what the finding says goes wrong, then read each candidate's meaning. The \
right kind is the one whose meaning is what the finding describes -- not the one \
that shares a word with it, and not the one that would be interesting if true.

If the finding describes something none of the candidates mean, say so. A name \
that does not fit is worse than no name: it is a wrong answer that reads like a \
right one, and the finding can be reported without a kind.

## How to answer

Reply with JSON only, and in this order:

{"reason": "...", "type": "..."}

- `reason` comes first and is where you work: one short clause saying what the \
finding describes, in the words of the kind you are about to choose.
- `type` comes last and must follow from it: one of the candidate names, or \
`none_of_these`.
"""


def schema(candidates: Sequence[str]) -> dict[str, Any]:
    """Reason first, name last.

    The order is the one the challenge stage was measured into: with the verdict
    first that stage voted and then justified, and eleven points of F1 turned on
    moving it. There is no reason to think a name behaves differently.
    """
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "type": {"type": "string", "enum": [*candidates, NONE]},
        },
        "required": ["reason", "type"],
        "additionalProperties": False,
    }


def build(finding: Mapping[str, Any], candidates: Sequence[str],
          catalogue: Mapping[str, str], excerpt: Sequence[str] = ()) -> str:
    body = [
        "# The finding",
        "",
        f"> {finding.get('title', '')}",
        "",
        f"In `{finding.get('file', '')}`, at line {finding.get('line', 0)}:",
        "",
        "```",
        finding.get("quote", "") or "(no line was quoted)",
        "```",
        "",
    ]
    if excerpt:
        body += ["Its surroundings:", "", "```", *excerpt, "```", ""]
    body += ["# The candidate kinds", ""]
    body += [f"- `{name}` -- {catalogue[name]}" for name in candidates]
    body += ["", f"- `{NONE}` -- none of the above describes what this finding says.", ""]
    return "\n".join(body)


def parse(text: str) -> tuple[str, str]:
    """``(type, reason)``. An unreadable answer names nothing.

    Failing to a name would be inventing one; failing to none leaves the finding
    reported without a kind, which is what the metric already knows how to score.
    """
    try:
        document = json.loads(text or "")
    except Exception:
        return "", "unparseable"
    if not isinstance(document, dict):
        return "", "answer was not an object"
    chosen = str(document.get("type", "") or "").strip()
    reason = str(document.get("reason", "") or "")
    return ("" if chosen == NONE else chosen), reason
