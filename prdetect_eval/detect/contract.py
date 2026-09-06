"""The response contract: what the model may say and how it is read back.

Constrained decoding guarantees the *shape* of the answer, never its content, so
everything here assumes a well-formed object that is still wrong: a line outside
the file, a filename from another repository, a quote that matches nothing. Such
a report is kept and counted, not silently dropped -- a detector that invents
locations must pay for it in the false-alarm column rather than disappear from
the numbers.

``quote`` exists so the deterministic filters of stage [5] have something to
check against; phase 0b only records it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

CONFIDENCE_FLOOR = 0.0
CONFIDENCE_CEILING = 1.0

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "quote": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["file", "line", "quote", "reason", "confidence"],
            },
        },
    },
    "required": ["findings"],
}


@dataclass(frozen=True)
class Report:
    """One defect the model claims, before it becomes a ``Prediction``."""

    file: str
    line: int
    quote: str
    reason: str
    confidence: float


@dataclass(frozen=True)
class Reject:
    """A response the harness could not turn into a report, and why."""

    stage: str
    detail: str
    payload: str


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return CONFIDENCE_FLOOR
    return max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, number))


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
            quote=str(item.get("quote", "")), reason=str(item.get("reason", "")),
            confidence=_clamp(item.get("confidence", 1.0)),
        ))
    return reports, rejects


def render(reports: list[Report]) -> str:
    """Serialise reports back into a contract-shaped response, for the stubs."""
    return json.dumps({"findings": [
        {"file": r.file, "line": r.line, "quote": r.quote, "reason": r.reason, "confidence": r.confidence}
        for r in reports
    ]}, ensure_ascii=False)
