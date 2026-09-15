"""The detector's answer: what the model may say and how it is read back.

An answer is a list of findings, each a file, a line, a kind, a short title, a
confidence and the quoted line it accuses. The kinds are compiled into the schema
as an `enum`, so constrained decoding cannot emit a kind the catalogue does not
have. The schema lists its fields in the order the model generates them.

Constrained decoding guarantees the shape and never the content: a well-formed
answer can still name a line past the end of the file. Those are kept and left to
the later checks rather than dropped here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Report:
    """One defect the model claims."""

    file: str
    line: int
    type: str
    title: str
    confidence: float
    # The accused line, copied by the model from its pack.
    quote: str = ""


@dataclass(frozen=True)
class Reject:
    """An answer that could not be read as findings, and why."""

    stage: str
    detail: str
    payload: str


ORDER = ("file", "line", "type", "title", "confidence", "quote")

FIELDS: dict[str, Any] = {
    "file": {"type": "string"},
    "line": {"type": "integer"},
    "title": {"type": "string"},
    "confidence": {"type": "number"},
    "quote": {"type": "string"},
}


def response_schema(types: Sequence[str]) -> dict[str, Any]:
    """The answer shape, with `type` limited to `types`."""
    properties: dict[str, Any] = {
        name: ({"type": "string", "enum": list(types)} if name == "type" else FIELDS[name])
        for name in ORDER
    }
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {"type": "object", "properties": properties, "required": list(ORDER)},
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
    """Turn one answer into reports, collecting what could not be read.

    A bad answer lands in the reject list instead of raising: one broken
    response must not stop a repository's run.
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
    """One report per line, keeping the most confident: two comments on one line are one comment."""
    best: dict[tuple[str, int], Report] = {}
    for report in reports:
        key = (report.file, report.line)
        if key not in best or report.confidence > best[key].confidence:
            best[key] = report
    return [report for report in reports if best.get((report.file, report.line)) is report]


def cap(reports: Sequence[Report], limit: int) -> list[Report]:
    """The most confident `limit` reports for one pull request; `limit` 0 keeps them all.

    A review that leaves dozens of comments on one pull request is unreadable
    whatever it found.
    """
    if limit <= 0:
        return list(reports)
    return sorted(reports, key=lambda report: -report.confidence)[:limit]


def render(reports: Sequence[Report]) -> str:
    """Reports back as an answer, for the stub."""
    return json.dumps({"findings": [
        {"file": r.file, "line": r.line, "type": r.type, "title": r.title,
         "confidence": r.confidence, **({"quote": r.quote} if r.quote else {})}
        for r in reports
    ]}, ensure_ascii=False)
