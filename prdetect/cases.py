"""A pull request as every stage sees it, and the files it is read from and written to.

A case is one pull request: its title and description, its diff, the lines it
adds, the files it changed and every other file at its source commit, and --
when the repository has an answer key -- its labels. Lines are one-based and
ranges include both ends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from prdetect.scoring.families import family

# Only source is reviewed.
CODE_SUFFIXES = (".py",)


def is_code(filename: str) -> bool:
    return filename.endswith(CODE_SUFFIXES)


@dataclass(frozen=True, order=True)
class Span:
    """A closed line range inside one repository-relative file."""

    file: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(f"invalid span {self.file}:{self.start_line}-{self.end_line}")

    @property
    def lines(self) -> range:
        return range(self.start_line, self.end_line + 1)

    def distance(self, other: "Span") -> int | None:
        """Line gap to `other`, 0 when they overlap, None when they are in different files."""
        if self.file != other.file:
            return None
        return max(0, other.start_line - self.end_line, self.start_line - other.end_line)


@dataclass(frozen=True)
class Label:
    """One defect the answer key records.

    `spans` holds the label's own location first and any equivalent location
    after it -- the other end of a defect that lives between two files. A finding
    at any of them is equally correct.
    """

    finding_id: str
    case_id: str
    type: str
    family: str
    in_scope: bool
    required: bool
    spans: tuple[Span, ...]
    title: str = ""
    cross_file: bool = False
    pure_deletion: bool = False
    rationale: str = ""

    @property
    def span(self) -> Span:
        return self.spans[0]

    @property
    def scored(self) -> bool:
        """Missing this label is a false negative; finding a label that is not scored is neutral."""
        return self.in_scope and self.required


@dataclass(frozen=True)
class Prediction:
    """One published finding."""

    case_id: str
    span: Span
    type: str
    confidence: float = 1.0
    detector: str = ""
    stage: str = ""
    message: str = ""

    @property
    def family(self) -> str:
        return family(self.type)


@dataclass
class Case:
    """One pull request."""

    case_id: str
    pr_title: str
    pr_description: str
    changed_files: tuple[str, ...]
    deleted_files: tuple[str, ...]
    added_lines: dict[str, frozenset[int]]
    head_files: dict[str, str]
    context_files: dict[str, str]
    diff: str
    labels: tuple[Label, ...] = ()
    is_defective: bool = False
    labelled: bool = False
    base_commit: str = ""
    head_commit: str = ""
    pull_request: dict = field(default_factory=dict)

    @property
    def scored_labels(self) -> tuple[Label, ...]:
        return tuple(label for label in self.labels if label.scored)

    @property
    def reviewable(self) -> bool:
        """Whether the pull request changes any source the detector is shown."""
        return any(is_code(name) for name in self.head_files)

    def touches(self, span: Span) -> bool:
        """True when the pull request adds or rewrites a line inside `span`."""
        return bool(self.added_lines.get(span.file, frozenset()).intersection(span.lines))

    def source_lines(self, filename: str) -> list[str]:
        return self.head_files.get(filename, "").split("\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error
    return rows


def _label(row: dict, case_id: str) -> Label:
    spans = [Span(row["file"], row["start_line"], row["end_line"])]
    for equivalent in row.get("equivalent_locations", []):
        spans.append(Span(equivalent["file"], equivalent["start_line"], equivalent["end_line"]))
    return Label(
        finding_id=row["finding_id"], case_id=case_id, type=row["type"], family=family(row["type"]),
        in_scope=bool(row["in_scope"]), required=bool(row["required"]), spans=tuple(spans),
        title=row.get("title", ""), cross_file=bool(row.get("cross_file")),
        pure_deletion=bool(row.get("pure_deletion")), rationale=row.get("rationale", ""),
    )


def load_cases(path: Path) -> list[Case]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Fetch it with: python -m prdetect.cli.fetch --repo {path.stem}")
    cases = []
    for row in read_jsonl(path):
        case_id = row["case_id"]
        cases.append(Case(
            case_id=case_id, pr_title=row["pr_title"], pr_description=row["pr_description"],
            changed_files=tuple(row["changed_files"]), deleted_files=tuple(row.get("deleted_files", [])),
            added_lines={name: frozenset(lines) for name, lines in row["added_lines"].items()},
            head_files=dict(row["head_files"]), context_files=dict(row.get("context_files", {})),
            diff=row["diff"], labels=tuple(_label(finding, case_id) for finding in row["findings"]),
            is_defective=bool(row.get("is_defective")), labelled=bool(row.get("labelled")),
            base_commit=row.get("base_commit", ""), head_commit=row.get("head_commit", ""),
            pull_request=dict(row.get("pull_request", {})),
        ))
    return cases


def load_predictions(path: Path) -> list[Prediction]:
    """A predictions file: one JSON object per finding."""
    predictions = []
    for row in read_jsonl(path):
        start = int(row.get("line", row.get("start_line")))
        predictions.append(Prediction(
            case_id=row["case_id"], span=Span(row["file"], start, int(row.get("end_line", start))),
            type=row.get("type", ""), confidence=float(row.get("confidence", 1.0)),
            detector=row.get("detector", ""), stage=row.get("stage", ""), message=row.get("message", ""),
        ))
    return predictions


def prediction_row(prediction: Prediction) -> dict:
    return {"case_id": prediction.case_id, "file": prediction.span.file,
            "line": prediction.span.start_line, "end_line": prediction.span.end_line,
            "type": prediction.type, "confidence": prediction.confidence,
            "detector": prediction.detector, "stage": prediction.stage, "message": prediction.message}
