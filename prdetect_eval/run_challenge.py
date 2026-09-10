"""Run stage [6] over a finished detector run.

Reads the predictions a run already wrote and asks the model, once per claim,
whether the lines it was shown refute it. Writes a new run directory holding the
survivors, so the result is scored by ``run_eval.py`` exactly like any other
prediction file and the two runs can be compared without re-detecting anything.

That separation is the point: the expensive pass stays on disk, and a change to
the challenge prompt costs twenty-six small calls rather than fifty large ones.
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
from detect import challenge as challenge_module
from detect import client as client_module
from detect import facts as facts_module

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Challenge the findings of a detector run.")
    parser.add_argument("--run", required=True, help="run id under runs/ holding predictions.jsonl")
    parser.add_argument("--dataset", help="defaults to the dataset recorded in the run")
    parser.add_argument("--model", help="Ollama model tag")
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL)
    parser.add_argument("--dry-run", action="store_true", help="write the prompts and stop")
    parser.add_argument("--radius", type=int, default=12, help="lines of context around a claim")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-id", help="defaults to <run>-challenged")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = args.out / args.run
    manifest = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset = args.dataset or manifest.get("dataset")
    if not dataset:
        raise SystemExit(f"{source}/config.json names no dataset; pass --dataset")
    cases = {case.case_id: case for case in load_cases(DATASETS / f"{dataset}.eval.jsonl")}
    claims = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines() if line]

    detector = None
    if not args.dry_run:
        if not args.model:
            raise SystemExit("give --model, or --dry-run")
        detector = client_module.OllamaClient(
            model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
            temperature=args.temperature, seed=args.seed, timeout=args.timeout, think=args.think,
        )
        problem = detector.health()
        if problem:
            raise SystemExit(f"{args.base_url}: {problem}")

    run_id = args.run_id or f"{args.run}-challenged"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    survivors, verdicts = [], []
    for index, claim in enumerate(claims, start=1):
        case = cases[claim["case_id"]]
        rows = challenge_module.excerpt(case, claim["file"], claim["line"], args.radius)
        counted = [fact.render() for fact in facts_module.collect(case)]
        user = challenge_module.build(claim, rows, counted)
        if detector is None:
            verdicts.append({**claim, "verdict": "stands", "quote": "", "reason": "dry run",
                             "excerpt_lines": len(rows), "user": user})
            survivors.append(claim)
            continue
        response = detector.complete(challenge_module.SYSTEM, user, challenge_module.VERDICT_SCHEMA)
        verdict, quote, reason = challenge_module.parse(response.text)
        # A refutation that quotes nothing in the excerpt is the same failure this
        # stage exists to catch, so it does not remove the finding.
        quoted = challenge_module.honours(quote, rows)
        settleable = challenge_module.settleable(claim["type"])
        honoured = verdict == "refuted" and quoted and settleable
        if not honoured:
            survivors.append(claim)
        verdicts.append({
            "case_id": claim["case_id"], "file": claim["file"], "line": claim["line"],
            "type": claim["type"], "title": claim["message"],
            "verdict": "refuted" if honoured else "stands",
            "claimed_verdict": verdict, "quote": quote, "reason": reason,
            "quote_found_in_excerpt": quoted,
            "excerpt_settles_the_kind": settleable,
            "excerpt_lines": len(rows), "error": response.error,
        })
        if not args.quiet:
            mark = "REFUTED" if honoured else "stands "
            print(f"[{index}/{len(claims)}] {mark} {claim['case_id']}:{claim['line']} -- {reason}",
                  file=sys.stderr, flush=True)

    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in survivors),
        encoding="utf-8", newline="\n")
    (run_dir / "verdicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts),
        encoding="utf-8", newline="\n")
    (run_dir / "system_prompt.txt").write_text(challenge_module.SYSTEM, encoding="utf-8", newline="\n")

    refuted = len(claims) - len(survivors)
    unhonoured = sum(1 for v in verdicts if v.get("claimed_verdict") == "refuted"
                     and not v.get("quote_found_in_excerpt"))
    unsettleable = sum(1 for v in verdicts if v.get("claimed_verdict") == "refuted"
                       and v.get("quote_found_in_excerpt")
                       and not v.get("excerpt_settles_the_kind"))
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": run_id, "stage": "challenge",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "challenged_run": args.run, "dataset": dataset,
        "claims": len(claims), "refuted": refuted, "survivors": len(survivors),
        "refutations_without_a_quote": unhonoured,
        "refutations_of_a_cross_file_claim": unsettleable,
        "radius": args.radius,
        "model": None if detector is None else detector.name,
        "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"\nclaims {len(claims)}  refuted {refuted}  survivors {len(survivors)}"
              f"  (rejected: {unhonoured} quoted nothing, {unsettleable} cross-file)")
        print(f"artefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
