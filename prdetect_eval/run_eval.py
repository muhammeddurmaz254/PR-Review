"""Evaluation entry point.

Every run writes its configuration, its raw predictions and its metrics under
``runs/<run_id>/``. Scoring is a pure function of those artefacts, so a changed
metric is re-scored with ``--rescore`` instead of re-running the detector -- on
a GPU pipeline that difference is hours.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence

import baselines as baseline_module
import metrics
import report as report_module
import steps as steps_module
from adapters import DEFAULT_EVAL, load_cases, load_predictions, write_predictions
from matching import match_all
from schema import Case, MatchConfig, Prediction

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def harness_commit() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"), cwd=HERE, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def baseline_summary(cases: Sequence[Case], config: MatchConfig) -> dict[str, dict]:
    """Score every trivial heuristic under the same configuration as the run."""
    summary = {}
    for name, build in baseline_module.BASELINES.items():
        predictions = build(cases)
        results = match_all(cases, predictions, config)
        card = metrics.score(cases, results)
        summary[name] = {
            **card.as_dict(),
            "pairwise": metrics.pairwise_accuracy(cases, results)["accuracy"],
            "fp_per_pr": round(card.false_alarm / len(cases), 4) if cases else 0.0,
            # The trade a detector actually makes is at the pull request, not at
            # the finding: `flag_everything` is the reference point, and it must
            # be printed beside the run rather than looked up separately.
            "pr_level": metrics.pr_level(cases, results).as_dict(),
            "leaky": name in baseline_module.LEAKY,
        }
    return summary


def resolve_predictions(args: argparse.Namespace, cases: Sequence[Case]) -> tuple[list[Prediction], str]:
    if args.baseline:
        if args.baseline not in baseline_module.BASELINES:
            raise SystemExit(
                f"unknown baseline {args.baseline!r}; choose from {', '.join(baseline_module.BASELINES)}"
            )
        return baseline_module.BASELINES[args.baseline](cases), f"baseline:{args.baseline}"
    if args.predictions:
        return load_predictions(args.predictions), str(args.predictions)
    return [], "empty"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score PR defect predictions against the corpus.")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="corpus JSON Lines file")
    parser.add_argument("--predictions", type=Path, help="prediction JSON Lines file to score")
    parser.add_argument("--baseline", help="score a trivial baseline instead of a prediction file")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="line slack around the label region; 0 means the report must land inside it")
    parser.add_argument("--type-mode", default="exact", choices=("exact", "family", "none"))
    parser.add_argument("--threshold", type=float, default=0.0, help="minimum confidence")
    parser.add_argument("--run-id", help="defaults to a UTC timestamp plus the source name")
    parser.add_argument("--out", type=Path, default=RUNS, help="directory holding run artefacts")
    parser.add_argument("--case-glob", action="append", default=[],
                        help="score only cases whose id matches (repeatable); for held-out splits")
    parser.add_argument("--no-baselines", action="store_true", help="skip the trivial baseline table")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--note", action="append", default=[], help="free-form note recorded with the run")
    args = parser.parse_args(argv)

    cases = load_cases(args.eval)
    if args.case_glob:
        cases = [case for case in cases if any(fnmatch(case.case_id, p) for p in args.case_glob)]
        if not cases:
            raise SystemExit(f"no case matches {args.case_glob}")
    config = MatchConfig(tolerance=args.tolerance, type_mode=args.type_mode)
    predictions, source = resolve_predictions(args, cases)
    # A prediction for a case outside the split is neither a hit nor a false
    # alarm here; dropping it keeps precision honest for the subset.
    kept = {case.case_id for case in cases}
    predictions = [prediction for prediction in predictions if prediction.case_id in kept]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = source.replace(":", "-").replace("/", "-").replace("\\", "-")[:40]
    run_id = args.run_id or f"{stamp}_{slug}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = metrics.evaluate(cases, predictions, config, args.threshold)
    trivial = None if args.no_baselines else baseline_summary(cases, config)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "corpus": str(args.eval),
        "cases": len(cases),
        "scored_labels": sum(len(case.scored_labels) for case in cases),
        "predictions": len(predictions),
        "case_glob": args.case_glob,
        "tolerance": args.tolerance,
        "type_mode": args.type_mode,
        "threshold": args.threshold,
        "harness_commit": harness_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        # Filled in once a model is in the loop; recorded here so a run is never
        # ambiguous about what produced it.
        "model": None, "quantization": None, "context_tokens": None,
        "prompt_version": None, "detectors": [], "sampling": None,
    }
    (run_dir / "config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_predictions(run_dir / "predictions.jsonl", predictions)
    (run_dir / "metrics.json").write_text(
        json.dumps({"config": manifest, "result": result, "baselines": trivial}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    markdown = report_module.render(run_id, manifest, result, trivial, notes=args.note)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    (run_dir / "step.md").write_text(
        steps_module.render(run_id, manifest, result, notes=args.note, baselines=trivial),
        encoding="utf-8", newline="\n"
    )

    if not args.quiet:
        print(report_module.console(result, trivial))
        print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
