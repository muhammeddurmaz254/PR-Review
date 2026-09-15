"""Publish a repository's findings: what the verifiers established, plus the rules.

    python -m prdetect.cli.publish --run <id>-named --agree <id>-located --rules <id>-rules --run-id <id>

With `--agree`, only claims both verify runs established are kept; then one claim
per site; then a rule finding is added only where nothing is published nearby, so
the model keeps the name of every site it covered; last, anything below
`--min-confidence` is dropped. Writes `predictions.jsonl` and `config.json`; the
report stage reads them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.cli import runs
from prdetect.detect import dedupe, verify


def _key(row: dict) -> tuple:
    return row["case_id"], row["file"], row["line"], row["type"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the verified findings plus the rules'.")
    parser.add_argument("--run", required=True, help="a verify run")
    parser.add_argument("--agree", help="a second verify run over the same claims: publish only what both established")
    parser.add_argument("--rules", required=True, help="a rules run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--radius", type=int, default=dedupe.RADIUS)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verified = runs.run_dir(args.run, args.out)
    manifest = runs.read_manifest(verified)
    agreement = None
    if args.agree:
        other_dir = runs.run_dir(args.agree, args.out)
        other = runs.read_manifest(other_dir)
        if other.get("verified_run") != manifest.get("verified_run"):
            raise SystemExit(f"--agree needs both runs to verify the same claims: {args.run} verified "
                             f"{manifest.get('verified_run')}, {args.agree} verified {other.get('verified_run')}")
        first = {_key(r): r for r in runs.read_rows(verified / "verdicts.jsonl")}
        second = {_key(r): r for r in runs.read_rows(other_dir / "verdicts.jsonl")}
        claims = runs.read_rows(runs.claims_file(runs.run_dir(manifest["verified_run"], args.out)))
        both = verify.agreed(claims, first, second)
        model = dedupe.one_per_site(both, args.radius)
        agreement = {"agree_run": other_dir.name, "claims": len(claims), "agreed": len(both)}
    else:
        model = runs.read_rows(verified / "predictions.jsonl")
    found = runs.read_rows(runs.run_dir(args.rules, args.out) / "predictions.jsonl")
    merged = dedupe.fill_gaps(model, found, args.radius)
    added = len(merged) - len(model)
    published = [row for row in merged if float(row.get("confidence", 1.0)) >= args.min_confidence]

    run_dir = args.out / args.run_id
    runs.write_rows(run_dir / "predictions.jsonl", published)
    runs.write_manifest(run_dir, {
        **manifest, "run_id": args.run_id, "stage": "publish", "model_run": verified.name,
        "rules_run": args.rules, "agreement": agreement, "predictions": len(published),
        "from_model": len(model), "from_rules": added, "rules_already_covered": len(found) - added,
        "radius": args.radius, "min_confidence": args.min_confidence,
        "below_min_confidence": len(merged) - len(published),
        **runs.provenance(argv),
    })
    if not args.quiet:
        print(f"{len(model)} from the model + {added} from the rules ({len(found) - added} already covered), "
              f"{len(merged) - len(published)} below {args.min_confidence} -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
