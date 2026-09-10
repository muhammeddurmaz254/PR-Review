"""Stage [4b]: name the findings of an open run.

Reads a detect run whose prompt carried no catalogue, puts a short list of
candidate kinds in front of each finding, and writes a prediction file with the
names filled in for ``run_eval.py``.

The point of the split is where breadth lives. A catalogue in the detector's
prompt was measured to cost 0.242 of F1 on halka_bench, eleven of eighteen false
alarms arriving under kinds the corpus does not contain. Here the catalogue can
be any size: it is searched, not read, and only the few kinds nearest a finding's
own words are ever shown.

    python run_detect.py --dataset halka --prompt-version review/open --run-id open
    python run_name.py --run open --model qwen3.8:27b --base-url ...
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import load_cases
from detect import client as client_module
from detect import naming

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Name the findings of an open detect run.")
    parser.add_argument("--run", required=True, help="run id under runs/ holding predictions.jsonl")
    parser.add_argument("--dataset", help="defaults to the dataset recorded in the run")
    parser.add_argument("--model", help="Ollama model tag")
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL)
    parser.add_argument("--pool", action="append", default=[],
                        help="datasets whose kinds go in the catalogue (default: all three)")
    parser.add_argument("--candidates", type=int, default=5,
                        help="how many kinds to show; five is where retrieval recall stops rising")
    parser.add_argument("--radius", type=int, default=6, help="lines of context around the quote")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-id", help="defaults to <run>-named")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = args.out / args.run
    manifest = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset = args.dataset or manifest.get("dataset")
    cases = {case.case_id: case for case in load_cases(DATASETS / f"{dataset}.eval.jsonl")}
    # A prediction row calls the finding's own words `message`, and does not
    # carry the quoted line at all -- that is in the anchors the detect run
    # wrote. Both are the query this stage searches with, so a mapping that
    # silently drops them makes every shortlist identical: the first run of this
    # stage did exactly that and scored 12% where retrieval alone allows 91%.
    quotes = {}
    anchors = source / "anchors.jsonl"
    if anchors.exists():
        for line in anchors.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                quotes[(row["case_id"], row["file"], row["line"])] = row.get("quote", "")
    findings = []
    for line in (source / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["title"] = row.get("message", "")
        row["quote"] = quotes.get((row["case_id"], row["file"], int(row.get("line") or 0)), "")
        findings.append(row)
    if not any(item["title"] for item in findings):
        raise SystemExit(f"{source}: no finding carries a title; the search would be blind")
    catalogue = naming.pool(tuple(args.pool) or ("halka", "demo_repo", "swrbench"))

    if not args.model:
        raise SystemExit("give --model")
    detector = client_module.OllamaClient(
        model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
        temperature=args.temperature, seed=args.seed, timeout=args.timeout, think=args.think,
    )
    problem = detector.health()
    if problem:
        raise SystemExit(f"{args.base_url}: {problem}")

    run_id = args.run_id or f"{args.run}-named"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    named, rows, failures, unnamed = [], [], 0, 0
    for index, finding in enumerate(findings, start=1):
        case = cases[finding["case_id"]]
        candidates = naming.rank(finding, catalogue, args.candidates)
        line = int(finding.get("line") or 0)
        source_lines = case.source_lines(finding["file"])
        excerpt = source_lines[max(0, line - 1 - args.radius):line + args.radius]
        user = naming.build(finding, candidates, catalogue, excerpt)
        response = detector.complete(naming.SYSTEM, user, naming.schema(candidates))
        failures += bool(response.error)
        chosen, reason = naming.parse(response.text)
        unnamed += not chosen
        named.append({k: v for k, v in finding.items()
                      if k not in ("title", "quote")} | {"type": chosen})
        rows.append({
            "case_id": finding["case_id"], "file": finding["file"], "line": line,
            "title": finding["title"], "quote": finding["quote"], "candidates": candidates,
            "type": chosen, "reason": reason, "error": response.error,
        })
        if not args.quiet:
            print(f"[{index}/{len(findings)}] {finding['case_id']}:{line} -> "
                  f"{chosen or '(none)'}  {reason[:60]}", file=sys.stderr, flush=True)

    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in named),
        encoding="utf-8", newline="\n")
    (run_dir / "names.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8", newline="\n")
    (run_dir / "system_prompt.txt").write_text(naming.SYSTEM, encoding="utf-8", newline="\n")
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": run_id, "stage": "name", "named_run": args.run,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalogue": len(catalogue), "candidates": args.candidates,
        "findings": len(findings), "unnamed": unnamed, "call_failures": failures,
        "complete": failures == 0, "model": detector.name,
        "model_build": detector.build() or None,
        "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"\nfindings {len(findings)}  catalogue {len(catalogue)}  "
              f"unnamed {unnamed}\nartefacts: {run_dir}")
    if failures:
        print(f"\n!! {failures} of {len(findings)} calls failed; this run is INCOMPLETE.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
