"""Merge what the model published with what the rules found.

    python run_publish.py --run v13c-zincir --rules rules-zincir --run-id final-zincir

The model's findings come first and keep their sites; a rule finding is added
only where nothing is published within the dedupe radius. D28 measured why
round the other way is wrong: on `crypto-03` the model had named the weak
digest at the line where `secret.literal` also fires, a plain union kept the
rule's name because its confidence is fixed higher, and a true finding became
a false alarm.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import claims_path
from detect import dedupe, verify
from run_detect import harness_commit

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the model's findings plus the rules'.")
    parser.add_argument("--run", required=True, help="the verify run")
    parser.add_argument("--rules", required=True, help="the run_rules.py run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--radius", type=int, default=dedupe.RADIUS)
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="drop what is published below this confidence, after agreement, "
                             "one-per-site and the rules -- the order measured in D33-D34")
    parser.add_argument("--agree", help="a second verify run over the same claims: publish only what "
                                        "both established (D32)")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads((args.out / args.run / "config.json").read_text(encoding="utf-8"))
    agreement = None
    if args.agree:
        other = json.loads((args.out / args.agree / "config.json").read_text(encoding="utf-8"))
        if other.get("verified_run") != manifest.get("verified_run"):
            raise SystemExit(f"--agree needs both runs to verify the same claims: {args.run} verified "
                             f"{manifest.get('verified_run')}, {args.agree} verified {other.get('verified_run')}")
        key = lambda r: (r["case_id"], r["file"], r["line"], r["type"])
        first = {key(r): r for r in _read(args.out / args.run / "verdicts.jsonl")}
        second = {key(r): r for r in _read(args.out / args.agree / "verdicts.jsonl")}
        claims = _read(claims_path(args.out / manifest["verified_run"]))
        both = verify.agreed(claims, first, second)
        # The order measured in D32: agreement first, then one comment per site.
        model = dedupe.one_per_site(both, args.radius)
        agreement = {"agree_run": args.agree, "claims": len(claims), "agreed": len(both)}
    else:
        model = _read(args.out / args.run / "predictions.jsonl")
    found = _read(args.out / args.rules / "predictions.jsonl")
    merged = dedupe.fill_gaps(model, found, args.radius)
    added = len(merged) - len(model)
    floored = [row for row in merged if float(row.get("confidence", 1.0)) >= args.min_confidence]
    below_floor = len(merged) - len(floored)
    merged = floored

    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8", newline="\n")
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": args.run_id, "stage": "publish", "model_run": args.run,
        "rules_run": args.rules, "predictions": len(merged),
        "from_model": len(model), "from_rules": added, "rules_dropped_as_duplicate": len(found) - added,
        "radius": args.radius, "agreement": agreement,
        "min_confidence": args.min_confidence, "below_floor": below_floor, "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness_commit": harness_commit(), "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"{len(model)} from the model + {added} from the rules "
              f"({len(found) - added} already covered) -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
