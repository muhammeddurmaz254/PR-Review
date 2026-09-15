"""Write one repository's analysis as a JSON report.

    python run_report.py --run <publish-run-id>

writes `reports/<repo>/<run-id>.json` at the project root: every pull request the
run reviewed, every finding it published -- file, line, kind, title, confidence --
and, when the repository has an answer key, the label each finding was assigned
to and the labels no finding reached. `report_txt.py` prints it as text. Nothing
is sent to Bitbucket.

A finding is assigned to a label the way the scorer assigns it: one-to-one, the
closest admissible pair first, in the right file within `--tolerance` lines.
Whether the kind and its family also agree is recorded on the assignment, and
the scores of all three rungs -- location, family, type -- go in the summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import metrics
from adapters import corpus_of, load_cases, load_predictions
from matching import match_all
from schema import Label, MatchConfig

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"
REPORTS = HERE.parent / "reports"
RUNGS = (("location", "none"), ("family", "family"), ("type", "exact"))


def _pull_requests(corpus: Path) -> dict[str, dict]:
    """Per case: the Bitbucket pull request it came from, and whether it is labelled."""
    out = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["case_id"]] = {**row.get("pull_request", {}), "labelled": row.get("labelled", True)}
    return out


def _label(label: Label) -> dict:
    return {"finding_id": label.finding_id, "type": label.type, "family": label.family,
            "file": label.span.file, "lines": [label.span.start_line, label.span.end_line],
            "title": label.title, "scored": label.scored}


def build(run_dir: Path, tolerance: int = 10) -> dict:
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    corpus = corpus_of(manifest, DATASETS)
    cases = load_cases(corpus)
    pulls = _pull_requests(corpus)
    predictions = load_predictions(run_dir / "predictions.jsonl")
    labelled = any(pull["labelled"] for pull in pulls.values())
    results = {name: match_all(cases, predictions, MatchConfig(tolerance=tolerance, type_mode=mode))
               for name, mode in RUNGS}

    by_case = defaultdict(list)
    for prediction in predictions:
        by_case[prediction.case_id].append(prediction)

    entries = []
    for case in cases:
        pull = pulls.get(case.case_id, {})
        located = results["location"][case.case_id]
        assigned = {id(match.prediction): match.label for match in located.matches}
        findings = []
        for prediction in sorted(by_case[case.case_id], key=lambda p: (p.span.file, p.span.start_line, p.type)):
            finding = {"file": prediction.span.file, "line": prediction.span.start_line,
                       "end_line": prediction.span.end_line, "type": prediction.type,
                       "family": prediction.family, "title": prediction.message,
                       "confidence": prediction.confidence, "detector": prediction.detector,
                       "stage": prediction.stage}
            if not pull.get("labelled", True):
                finding["verdict"] = "unscored"
            else:
                label = assigned.get(id(prediction))
                if label is None:
                    finding["verdict"] = "false_alarm"
                    finding["assigned_to"] = None
                else:
                    finding["verdict"] = "true_positive" if label.scored else "neutral"
                    finding["assigned_to"] = {**_label(label),
                                              "same_type": label.type == prediction.type,
                                              "same_family": label.family == prediction.family}
            findings.append(finding)
        entries.append({
            "pr_id": pull.get("id"), "case_id": case.case_id, "title": case.pr_title,
            "url": pull.get("url"), "source_branch": pull.get("source_branch") or case.branch,
            "head_commit": case.head_commit, "changed_files": list(case.changed_files),
            "labelled": pull.get("labelled", True),
            "findings": findings,
            "missed": [_label(label) for label in located.unmatched_labels if label.scored],
        })

    summary = {"pull_requests": len(cases), "findings": len(predictions),
               "pull_requests_with_findings": sum(bool(entry["findings"]) for entry in entries)}
    if labelled:
        summary["scores"] = {name: metrics.score(cases, results[name]).as_dict() for name, _ in RUNGS}
    return {
        "repository": manifest.get("dataset"), "run": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(corpus), "prompt_version": manifest.get("prompt_version"),
        "model": manifest.get("model"), "tolerance": tolerance, "labelled": labelled,
        "summary": summary, "pull_requests": entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one repository's analysis as a JSON report.")
    parser.add_argument("--run", required=True, help="run id under runs/, normally the publish run")
    parser.add_argument("--tolerance", type=int, default=10, help="lines a finding may sit from its label")
    parser.add_argument("--out", type=Path, help="defaults to reports/<repo>/<run>.json")
    args = parser.parse_args(argv)

    run_dir = RUNS / args.run
    if not (run_dir / "predictions.jsonl").exists():
        raise SystemExit(f"{run_dir} has no predictions.jsonl")
    report = build(run_dir, args.tolerance)
    out = args.out or REPORTS / report["repository"] / f"{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"{report['repository']}: {summary['pull_requests']} pull requests, {summary['findings']} findings -> {out}")
    for name, card in summary.get("scores", {}).items():
        print(f"  {name:<8} TP={card['tp']} FP={card['fp']} FN={card['fn']}  F1={card['f1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
