"""Run the verify stages over a finished detector run.

Reads the predictions a run already wrote and asks the model, once per claim,
either of two questions:

``--stage refute`` (stage [6a], the default)
    Do the lines shown contradict the claim? Catches a report the prompt already
    denies.

``--stage consequence`` (stage [6b])
    Name the run in which this goes wrong. Catches a report that is *true* about
    code that is *correct*, which is what [6a] measurably cannot do.

``--stage both`` asks both and removes a claim either answer removes, at two
small calls per claim.

Either way a new run directory holds the survivors, so the result is scored by
``run_eval.py`` exactly like any other prediction file and the runs can be
compared without re-detecting anything. That separation is the point: the
expensive pass stays on disk, and a change to a verify prompt costs sixty small
calls rather than a hundred and ten large ones.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import load_cases, eval_path
from detect import challenge as challenge_module
from detect import client as client_module
from detect import consequence as consequence_module
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
    parser.add_argument("--stage", choices=("refute", "consequence", "both"), default="refute",
                        help="which verify question to ask; see the module docstring")
    parser.add_argument("--strict-consequence", action="store_true",
                        help="stage [6b]: also remove a claim of harm that names no trigger")
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
    cases = {case.case_id: case for case in load_cases(eval_path(dataset, DATASETS))}
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

    ask_refute = args.stage in ("refute", "both")
    ask_consequence = args.stage in ("consequence", "both")

    survivors, verdicts, harms = [], [], []
    for index, claim in enumerate(claims, start=1):
        case = cases[claim["case_id"]]
        rows = challenge_module.excerpt(case, claim["file"], claim["line"], args.radius)
        counted = [fact.render() for fact in facts_module.collect(case)]
        marks: list[str] = []
        refuted = removed = False

        if ask_refute:
            user = challenge_module.build(claim, rows, counted)
            if detector is None:
                verdicts.append({**claim, "verdict": "stands", "quote": "", "reason": "dry run",
                                 "excerpt_lines": len(rows), "user": user})
            else:
                response = detector.complete(
                    challenge_module.SYSTEM, user, challenge_module.VERDICT_SCHEMA)
                verdict, quote, reason = challenge_module.parse(response.text)
                # A refutation that quotes nothing in the excerpt is the same
                # failure this stage exists to catch, so it does not remove the
                # finding.
                quoted = challenge_module.honours(quote, rows)
                settleable = challenge_module.settleable(claim["type"])
                refuted = verdict == "refuted" and quoted and settleable
                verdicts.append({
                    "case_id": claim["case_id"], "file": claim["file"], "line": claim["line"],
                    "type": claim["type"], "title": claim["message"],
                    "verdict": "refuted" if refuted else "stands",
                    "claimed_verdict": verdict, "quote": quote, "reason": reason,
                    "quote_found_in_excerpt": quoted,
                    "excerpt_settles_the_kind": settleable,
                    "excerpt_lines": len(rows), "error": response.error,
                })
                marks.append("REFUTED" if refuted else "stands")
                removed = removed or refuted

        if ask_consequence:
            user = consequence_module.build(claim, rows, counted)
            if detector is None:
                harms.append({**claim, "harm": "dry run", "trigger": "", "consequence": "",
                              "excerpt_lines": len(rows), "user": user})
            else:
                response = detector.complete(
                    consequence_module.SYSTEM, user, consequence_module.HARM_SCHEMA)
                harm, trigger, effect = consequence_module.parse(response.text)
                # Both readings are stored. The lenient one removes a claim only
                # when the model names the harm as absent; the strict one also
                # removes a claim of harm that could not name a run. Storing both
                # is what lets run_regate.py score the second without a GPU pass.
                grounded = consequence_module.concrete(trigger, effect)
                harmless = consequence_module.harmless(harm)
                vacuous = not harmless and not grounded
                dropped_here = harmless or (args.strict_consequence and vacuous)
                harms.append({
                    "case_id": claim["case_id"], "file": claim["file"], "line": claim["line"],
                    "type": claim["type"], "title": claim["message"],
                    "harm": harm, "trigger": trigger, "consequence": effect,
                    "harmless": harmless, "named_a_run": grounded,
                    "claims_harm_without_a_run": vacuous,
                    "excerpt_lines": len(rows), "error": response.error,
                })
                marks.append(f"harm={harm or '?'}")
                removed = removed or dropped_here

        if detector is None or not removed:
            survivors.append(claim)
        if not args.quiet and detector is not None:
            mark = "DROPPED" if removed else "keeps  "
            print(f"[{index}/{len(claims)}] {mark} {claim['case_id']}:{claim['line']}"
                  f"  [{' '.join(marks)}]", file=sys.stderr, flush=True)

    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in survivors),
        encoding="utf-8", newline="\n")
    if verdicts:
        (run_dir / "verdicts.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts),
            encoding="utf-8", newline="\n")
    if harms:
        (run_dir / "consequences.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in harms),
            encoding="utf-8", newline="\n")
    # One file per question asked, named after the question, so a reader of the
    # run directory can tell which node produced which removal.
    if ask_refute:
        (run_dir / "system_prompt.refute.txt").write_text(
            challenge_module.SYSTEM, encoding="utf-8", newline="\n")
    if ask_consequence:
        (run_dir / "system_prompt.consequence.txt").write_text(
            consequence_module.SYSTEM, encoding="utf-8", newline="\n")

    removed = len(claims) - len(survivors)
    unhonoured = sum(1 for v in verdicts if v.get("claimed_verdict") == "refuted"
                     and not v.get("quote_found_in_excerpt"))
    unsettleable = sum(1 for v in verdicts if v.get("claimed_verdict") == "refuted"
                       and v.get("quote_found_in_excerpt")
                       and not v.get("excerpt_settles_the_kind"))
    refuted = sum(1 for v in verdicts if v.get("verdict") == "refuted")
    harmless = sum(1 for h in harms if h.get("harmless"))
    vacuous = sum(1 for h in harms if h.get("claims_harm_without_a_run"))
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": run_id, "stage": f"verify:{args.stage}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "challenged_run": args.run, "dataset": dataset,
        "verify_stage": args.stage,
        "strict_consequence": bool(args.strict_consequence),
        "claims": len(claims), "removed": removed, "survivors": len(survivors),
        "refuted": refuted,
        "refutations_without_a_quote": unhonoured,
        "refutations_of_a_cross_file_claim": unsettleable,
        "named_no_harm": harmless,
        "claimed_harm_without_a_run": vacuous,
        "radius": args.radius,
        "model": None if detector is None else detector.name,
        "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"\nclaims {len(claims)}  removed {removed}  survivors {len(survivors)}")
        if ask_refute:
            print(f"  [6a] refuted {refuted}"
                  f"  (rejected: {unhonoured} quoted nothing, {unsettleable} cross-file)")
        if ask_consequence:
            print(f"  [6b] named no harm {harmless}"
                  f"  (claimed harm but named no run: {vacuous}"
                  f"{', applied' if args.strict_consequence else ', stored only'})")
        print(f"artefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
