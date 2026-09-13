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

from detect import dedupe
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
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    model = _read(args.out / args.run / "predictions.jsonl")
    found = _read(args.out / args.rules / "predictions.jsonl")
    merged = dedupe.fill_gaps(model, found, args.radius)
    added = len(merged) - len(model)

    manifest = json.loads((args.out / args.run / "config.json").read_text(encoding="utf-8"))
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8", newline="\n")
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": args.run_id, "stage": "publish", "model_run": args.run,
        "rules_run": args.rules, "predictions": len(merged),
        "from_model": len(model), "from_rules": added, "rules_dropped_as_duplicate": len(found) - added,
        "radius": args.radius, "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness_commit": harness_commit(), "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"{len(model)} from the model + {added} from the rules "
              f"({len(found) - added} already covered) -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
