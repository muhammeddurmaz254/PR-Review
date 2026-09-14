"""Run the deterministic rules over a corpus.

    python run_rules.py --dataset zincir --run-id rules-zincir

Writes a run directory shaped like a detector's, so `run_verify.py` and
`run_eval.py` read it the same way. No model, no network: every finding is a
fact about the change -- a value added, a line deleted -- so two runs of this
command on one corpus are identical.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import eval_path, load_cases
from detect import prompt, rules
from run_detect import harness_commit

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic rules for config and test files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    corpus = eval_path(args.dataset, DATASETS)
    cases = load_cases(corpus)
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    found = []
    for case in cases:
        for finding in rules.run(case):
            found.append({"case_id": finding.case_id, "file": finding.file, "line": finding.line,
                          "quote": finding.quote, "type": finding.type, "message": finding.message,
                          "confidence": finding.confidence, "detector": f"rules.{finding.rule}"})
            if not args.quiet:
                print(f"{finding.case_id}: {finding.file}:{finding.line} {finding.type} "
                      f"[{finding.rule}]", file=sys.stderr)

    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in found),
        encoding="utf-8", newline="\n")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "rules", "dataset": args.dataset, "corpus": str(corpus),
        "cases": len(cases), "predictions": len(found),
        "rules": [rule.__name__ for rule in rules.RULES],
        # The names are the catalogue's, and the stages downstream read the
        # definitions from it: a rule finding is verified against the same words
        # as the model's. Nothing here calls the model.
        "prompt_version": prompt.PROMPT_VERSION, "deletions": True, "facts": False,
        "detectors": ["rules"], "model": None, "sampling": None,
        "types": sorted({row["type"] for row in found}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness_commit": harness_commit(), "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"{len(found)} findings -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
