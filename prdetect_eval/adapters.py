"""Loaders that turn stored artefacts into canonical objects.

Scoring must be reproducible from disk alone, so nothing here reaches for the
built git repository: ``eval.jsonl`` already carries the diff, the post-change
file contents and the touched line numbers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from schema import FAMILY_BY_TYPE, Case, Distractor, Label, Prediction, Span

DEFAULT_EVAL = Path(__file__).resolve().parents[1] / "repo" / "eval.jsonl"


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error


def _label(row: dict, case_id: str) -> Label:
    spans = [Span(row["file"], row["start_line"], row["end_line"])]
    for equivalent in row.get("equivalent_locations", []):
        spans.append(Span(equivalent["file"], equivalent["start_line"], equivalent["end_line"]))
    return Label(
        finding_id=row["finding_id"], case_id=case_id, type=row["type"],
        family=row.get("family") or FAMILY_BY_TYPE.get(row["type"], ""),
        in_scope=bool(row["in_scope"]), required=bool(row["required"]), role=row.get("role", ""),
        spans=tuple(spans), anchor_rule=row.get("anchor_rule", ""), anchor_text=row.get("anchor_text", ""),
        in_diff=bool(row.get("in_diff", True)), severity=row.get("severity", ""), cwe=row.get("cwe", ""),
    )


def load_cases(path: Path = DEFAULT_EVAL) -> list[Case]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run python tools/export_eval.py in repo/.")
    cases = []
    for row in _read_jsonl(path):
        if "added_lines" not in row or "head_files" not in row:
            raise SystemExit(
                f"{path} predates the harness. Re-run python tools/export_eval.py in repo/."
            )
        case_id = row["case_id"]
        cases.append(Case(
            case_id=case_id, pair_id=row["pair_id"], variant=row["variant"], difficulty=row["difficulty"],
            primary_type=row["primary_type"], is_defective=bool(row["is_defective"]),
            pr_title=row["pr_title"], pr_description=row["pr_description"],
            changed_files=tuple(row["changed_files"]), noise_files=tuple(row.get("noise_files", [])),
            deleted_files=tuple(row.get("deleted_files", [])),
            added_lines={name: frozenset(lines) for name, lines in row["added_lines"].items()},
            head_files=dict(row["head_files"]), diff=row["diff"],
            labels=tuple(_label(finding, case_id) for finding in row["findings"]),
            distractors=tuple(
                Distractor(Span(d["file"], d["start_line"], d["end_line"]), d.get("looks_like", ""), d.get("why_not", ""))
                for d in row.get("distractors", [])
            ),
            base_commit=row.get("base_commit", ""), head_commit=row.get("head_commit", ""),
            branch=row.get("branch", ""),
        ))
    return cases


def load_predictions(path: Path) -> list[Prediction]:
    """Read a prediction file: one JSON object per reported defect."""
    predictions = []
    for row in _read_jsonl(path):
        start = int(row.get("line", row.get("start_line")))
        predictions.append(Prediction(
            case_id=row["case_id"], span=Span(row["file"], start, int(row.get("end_line", start))),
            type=row.get("type", ""), confidence=float(row.get("confidence", 1.0)),
            detector=row.get("detector", ""), stage=row.get("stage", ""), message=row.get("message", ""),
        ))
    return predictions


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps({
                "case_id": prediction.case_id, "file": prediction.span.file,
                "line": prediction.span.start_line, "end_line": prediction.span.end_line,
                "type": prediction.type, "confidence": prediction.confidence,
                "detector": prediction.detector, "stage": prediction.stage, "message": prediction.message,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def group_by_case(items: Iterable) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(item.case_id, []).append(item)
    return grouped
