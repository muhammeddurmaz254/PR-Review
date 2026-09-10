"""The response contract: what the model may say and how it is read back.

The output is deliberately small -- type, file, line, one short title. The task
is to find the defect and name its kind, not to write the review comment, so
anything longer costs tokens and invites the model to argue itself into a
finding it does not have.

The type list is compiled into the schema as an ``enum``, so constrained
decoding cannot emit a class the dataset does not have. That matters for the
metric: type accuracy then measures judgment rather than formatting.

Constrained decoding guarantees the shape and never the content, so a
well-formed answer can still be wrong -- a line past the end of the file, a
filename from another project. Those are kept and counted, not dropped: a
detector that invents locations must pay for it in the false-alarm column
rather than disappear from the numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Report:
    """One defect the model claims, before it becomes a ``Prediction``."""

    file: str
    line: int
    type: str
    title: str
    confidence: float
    # The line the model says it is accusing, copied from the prompt. Empty
    # under prompt versions that do not ask for it.
    quote: str = ""


@dataclass(frozen=True)
class Reject:
    """A response the harness could not turn into a report, and why."""

    stage: str
    detail: str
    payload: str


# The order a schema lists its properties is the order constrained decoding
# emits them, and with `think: false` it is the only place the model works. The
# original order put `quote` last, after the file, the line, the type, the title
# and even the confidence -- so the "line it accuses" was written *after* the
# claim was already committed, and could only ever be retro-fitted to it. That is
# a plausible reason stage [5] never rejected anything: the model copies whatever
# sits at the line it had already chosen.
#
# The same fault, found and measured in the challenge stage, cost that stage
# eleven points of F1. LEGACY is kept because five prompt versions were measured
# under it and their numbers mean nothing under another order.
LEGACY_ORDER = ("file", "line", "type", "title", "confidence", "quote")
# The same order with the name removed: an open run reports what is wrong and
# stage [4b] decides what it is called.
OPEN_ORDER = ("file", "line", "title", "confidence", "quote")
EVIDENCE_ORDER = ("quote", "title", "type", "file", "line", "confidence")
# Measured against EVIDENCE_ORDER: putting the quote first cost six findings and
# four points of F1. A quote is a copy, not a place to think, so leading with it
# only forces an early commitment to a line -- the model answers when it is
# already sure. What won in the challenge was the *reason* before the vote, and
# the detector's reason is `title`.
CLAIM_ORDER = ("title", "quote", "type", "file", "line", "confidence")

FIELDS: dict[str, Any] = {
    "file": {"type": "string"},
    "line": {"type": "integer"},
    "title": {"type": "string"},
    "confidence": {"type": "number"},
    "quote": {"type": "string"},
}


def response_schema(types: Sequence[str], quote: bool = False,
                    order: Sequence[str] = LEGACY_ORDER) -> dict[str, Any]:
    """The answer shape, in the order the model is to produce it.

    ``quote`` adds the accused line, which stage [5] checks. Constrained decoding
    can force the field to exist; only the order decides whether it was read or
    invented.
    """
    names = [name for name in order if quote or name != "quote"]
    properties: dict[str, Any] = {
        name: ({"type": "string", "enum": list(types)} if name == "type" else FIELDS[name])
        for name in names
    }
    required = list(names)
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {"type": "object", "properties": properties, "required": required},
            },
        },
        "required": ["findings"],
    }


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def parse(payload: str) -> tuple[list[Report], list[Reject]]:
    """Turn one model response into reports, collecting what could not be read.

    A server without constrained decoding, or one that fell back to free text on
    a truncated generation, lands in the reject list instead of raising: a single
    bad response must not abort a corpus run.
    """
    text = (payload or "").strip()
    if not text:
        return [], [Reject("empty", "model returned no text", "")]
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        return [], [Reject("json", str(error), text[:500])]
    if not isinstance(document, dict):
        return [], [Reject("shape", f"expected an object, got {type(document).__name__}", text[:500])]

    raw = document.get("findings")
    if raw is None:
        return [], [Reject("shape", "no 'findings' key", text[:500])]
    if not isinstance(raw, list):
        return [], [Reject("shape", f"'findings' is {type(raw).__name__}, not a list", text[:500])]

    reports: list[Report] = []
    rejects: list[Reject] = []
    for item in raw:
        if not isinstance(item, dict):
            rejects.append(Reject("item", "finding is not an object", json.dumps(item)[:200]))
            continue
        try:
            line = int(item["line"])
        except (KeyError, TypeError, ValueError):
            rejects.append(Reject("line", "missing or non-integer line", json.dumps(item)[:200]))
            continue
        if line < 1:
            rejects.append(Reject("line", f"line {line} is not a file position", json.dumps(item)[:200]))
            continue
        filename = str(item.get("file", "")).strip().replace("\\", "/")
        if not filename:
            rejects.append(Reject("file", "missing file", json.dumps(item)[:200]))
            continue
        reports.append(Report(
            file=filename, line=line,
            type=str(item.get("type", "")).strip(),
            title=str(item.get("title", "")).strip(),
            confidence=_clamp(item.get("confidence", 1.0)),
            quote=str(item.get("quote", "") or ""),
        ))
    return reports, rejects


def dedupe(reports: Sequence[Report]) -> list[Report]:
    """One report per line, keeping the most confident.

    The justification is the product rule, not the measurement: a review bot that
    leaves two comments on one line is worse to read than one, whatever it found.
    That can be argued before seeing any data, which is what separates it from a
    rule fitted to the answers.

    The measurement only checks the price. Across 188 cases of three corpora it
    fires exactly once -- on a line the model named twice under two types, one
    matching the label and one not -- and it removes no true positive anywhere.
    So the gain is a single false alarm, which is noise; the reason to keep it is
    that it costs nothing and the alternative is a worse comment thread.

    The cost it *could* have: two genuinely different defects on one line become
    one report. None occurs in these corpora, and a reviewer told about the line
    will see both.
    """
    best: dict[tuple[str, int], Report] = {}
    for report in reports:
        key = (report.file, report.line)
        if key not in best or report.confidence > best[key].confidence:
            best[key] = report
    return [report for report in reports if best.get((report.file, report.line)) is report]


def cap(reports: Sequence[Report], limit: int) -> list[Report]:
    """The most confident ``limit`` reports for one pull request.

    Measured on SWRBench: prompt v4 returned sixty-five findings, forty-three of
    them from the two largest packs, and localization was ten of twenty-five at
    every cap from one to unlimited. Past the first finding the model adds volume
    and no signal, so this is a product constraint with nothing to lose -- a
    review bot that posts thirty-two comments on one pull request is unusable
    whatever its recall.

    The default sits above the corpus's own density of one labelled finding per
    defective pull request, so it cannot be mistaken for tuning against it.
    """
    if limit <= 0:
        return list(reports)
    return sorted(reports, key=lambda report: -report.confidence)[:limit]


def render(reports: Sequence[Report]) -> str:
    """Serialise reports back into a contract-shaped response, for the stubs."""
    return json.dumps({"findings": [
        {"file": r.file, "line": r.line, "type": r.type, "title": r.title,
         "confidence": r.confidence, **({"quote": r.quote} if r.quote else {})}
        for r in reports
    ]}, ensure_ascii=False)
