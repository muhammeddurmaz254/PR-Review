"""Loaders that turn stored artefacts into canonical objects.

Scoring must be reproducible from disk alone, so nothing here reaches for the
built git repository: ``eval.jsonl`` already carries the diff, the post-change
file contents and the touched line numbers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from schema import FAMILY_BY_TYPE, Case, Distractor, Label, Prediction, Span, catalog_family

DEFAULT_EVAL = Path(__file__).resolve().parent / "datasets" / "demo_repo.eval.jsonl"
def eval_path(dataset: str, datasets_dir: Path) -> Path:
    """The eval file a dataset name refers to.

    `zincir_bench` ships two of them and only one is open (B9). Naming the
    dataset and not the file gets `dev`; reaching `holdout` has to be typed out
    and written down in the corpus README. That rule lived in `run_detect.py`
    alone, so `run_challenge.py` and `run_regate.py` went looking for a
    `zincir.eval.jsonl` that does not exist -- one stage of the pipeline knew
    about the split and the next two did not.
    """
    if dataset == "zincir":
        return datasets_dir / "zincir_dev.eval.jsonl"
    return datasets_dir / f"{dataset}.eval.jsonl"


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
    focus = None
    if row.get("focus_start"):
        focus = Span(row["file"], int(row["focus_start"]), int(row.get("focus_end") or row["focus_start"]))
    return Label(
        finding_id=row["finding_id"], case_id=case_id, type=row["type"],
        # The catalogue wins when it knows the type. Both sides of a family
        # match have to speak one vocabulary: the prediction's family comes from
        # ``FAMILY_BY_TYPE``, so a label carrying its *case's* family instead --
        # halka files three data-layer types under "correctness" that way --
        # makes the rung disagree with itself. Corpora outside the catalogue
        # (demo_repo, SWRBench) keep the family they ship.
        family=catalog_family(row["type"]) or row.get("family") or FAMILY_BY_TYPE.get(row["type"], ""),
        in_scope=bool(row["in_scope"]), required=bool(row["required"]), role=row.get("role", ""),
        spans=tuple(spans), anchor_rule=row.get("anchor_rule", ""), anchor_text=row.get("anchor_text", ""),
        in_diff=bool(row.get("in_diff", True)), severity=row.get("severity", ""), cwe=row.get("cwe", ""),
        focus=focus, title=row.get("title", ""),
        cross_file=bool(row.get("cross_file")), pure_deletion=bool(row.get("pure_deletion")),
        # Section 0.1. These five were written by the exporters and dropped
        # here, which made every measurement that needed them -- grounding
        # class, product-facing scope, the rule-id join -- impossible to run
        # from the loaded corpus. ``file_role`` is section 0.2: read it, never
        # infer it from the path.
        file_role=row.get("file_role", ""),
        grounding=row.get("grounding", ""),
        rule_ids=tuple(row.get("rule_ids", ())),
        product_scope=bool(row.get("product_scope")),
        product_match_class=row.get("product_match_class", ""),
        rationale=row.get("rationale", ""),
    )


def load_cases(path: Path = DEFAULT_EVAL) -> list[Case]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Build it with python datasets/build_<dataset>.py")
    cases = []
    for row in _read_jsonl(path):
        if "added_lines" not in row or "head_files" not in row:
            raise SystemExit(f"{path} predates the harness. Rebuild it with datasets/build_<dataset>.py")
        case_id = row["case_id"]
        cases.append(Case(
            case_id=case_id, pair_id=row["pair_id"], variant=row["variant"], difficulty=row["difficulty"],
            primary_type=row["primary_type"], is_defective=bool(row["is_defective"]),
            pr_title=row["pr_title"], pr_description=row["pr_description"],
            changed_files=tuple(row["changed_files"]), noise_files=tuple(row.get("noise_files", [])),
            deleted_files=tuple(row.get("deleted_files", [])),
            added_lines={name: frozenset(lines) for name, lines in row["added_lines"].items()},
            head_files=dict(row["head_files"]),
            context_files=dict(row.get("context_files", {})), diff=row["diff"],
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
