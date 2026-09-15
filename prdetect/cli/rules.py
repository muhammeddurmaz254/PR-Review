"""Run the deterministic rules over a repository's pull requests.

    python -m prdetect.cli.rules --repo genis_olcum_reposu --run-id <id>-rules

No model and no network: every finding is a fact about a change, so two runs on
the same cases are identical. Writes `predictions.jsonl` and `config.json` under
`runs/<run-id>/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.cases import load_cases
from prdetect.cli import runs
from prdetect.detect import rules


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic rules over a repository's pull requests.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cases", type=Path, help="a cases file other than data/cases/<repo>.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases_path = args.cases or paths.cases_file(args.repo)
    cases = load_cases(cases_path)
    run_dir = args.out / args.run_id

    found = []
    for case in cases:
        for finding in rules.run(case):
            found.append({"case_id": finding.case_id, "file": finding.file, "line": finding.line,
                          "quote": finding.quote, "type": finding.type, "message": finding.message,
                          "confidence": finding.confidence, "detector": f"rules.{finding.rule}"})
            if not args.quiet:
                print(f"{finding.case_id}: {finding.file}:{finding.line} {finding.type} [{finding.rule}]",
                      file=sys.stderr)

    runs.write_rows(run_dir / "predictions.jsonl", found)
    runs.write_manifest(run_dir, {
        "run_id": args.run_id, "stage": "rules", "repository": args.repo, "cases_file": str(cases_path),
        "cases": len(cases), "predictions": len(found), "rules": [rule.__name__ for rule in rules.RULES],
        **runs.provenance(argv),
    })
    if not args.quiet:
        print(f"{len(found)} findings -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
