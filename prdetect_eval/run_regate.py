"""Re-apply the post-model stages to a finished run, without a server.

The harness's rule is that a score is a pure function of the artefacts a run
left on disk, so a change to a filter should not cost a second GPU pass. This
tool is that rule made usable: it reads the answers a detect run stored and the
verdicts a challenge run stored, applies stage [5b] (the evidence gate) and the
cross-file rule of stage [6] as they stand in the code today, and writes a new
prediction file for ``run_eval.py``.

It cannot re-ask the model, so it cannot measure a prompt change. What it does
measure is exactly what these two stages are: deterministic filters over reports
that already exist.

    python run_regate.py --run halka-v6 --challenge halka-v6-challenged-v3 \
        --run-id halka-v6-regated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from adapters import claims_path, eval_path, load_cases
from detect import challenge as challenge_module
from detect import contract
from detect import consequence as consequence_module
from detect import evidence as evidence_module
from detect import scope as scope_module

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"


def _report(row: dict) -> contract.Report:
    return contract.Report(
        file=row["file"], line=row["line"], type=row["type"],
        title=row.get("message", ""), confidence=float(row.get("confidence", 1.0)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="detect run id under runs/")
    parser.add_argument("--challenge", help="verify run id whose verdicts.jsonl to replay")
    parser.add_argument("--consequence", help="verify run id whose consequences.jsonl to replay")
    parser.add_argument("--strict-consequence", action="store_true",
                        help="also drop a claim of harm that named no run")
    parser.add_argument("--dataset", help="defaults to the dataset recorded in the run")
    parser.add_argument("--scope-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cross-file-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--one-per-line", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", help="defaults to <run>-regated")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = args.out / args.run
    manifest = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset = args.dataset or manifest.get("dataset")
    cases = {c.case_id: c for c in load_cases(eval_path(dataset, DATASETS))}
    claims_file = claims_path(source)
    claims = [json.loads(l) for l in claims_file.read_text().splitlines() if l.strip()]

    counts = {"claims": len(claims), "named_no_harm": 0, "claimed_harm_without_a_run": 0,
              "out_of_scope": 0, "off_operation": 0, "refuted": 0,
              "spared_cross_file": 0, "spared_unquoted": 0, "deduped": 0}
    survivors = list(claims)

    if args.challenge:
        verdicts = [json.loads(l) for l in
                    (args.out / args.challenge / "verdicts.jsonl").read_text().splitlines() if l.strip()]
        if len(verdicts) != len(claims):
            raise SystemExit(
                f"{args.challenge} holds {len(verdicts)} verdicts for {len(claims)} claims; "
                f"it did not challenge this run.")
        survivors = []
        for claim, verdict in zip(claims, verdicts):
            if verdict.get("claimed_verdict") != "refuted":
                survivors.append(claim)
                continue
            if not verdict.get("quote_found_in_excerpt"):
                counts["spared_unquoted"] += 1
                survivors.append(claim)
                continue
            if args.cross_file_gate and not challenge_module.settleable(claim["type"]):
                counts["spared_cross_file"] += 1
                survivors.append(claim)
                continue
            counts["refuted"] += 1

    if args.consequence:
        answers = [json.loads(l) for l in
                   (args.out / args.consequence / "consequences.jsonl").read_text().splitlines()
                   if l.strip()]
        by_claim = {(a["case_id"], a["file"], a["line"], a["type"]): a for a in answers}
        kept = []
        for claim in survivors:
            answer = by_claim.get((claim["case_id"], claim["file"], claim["line"], claim["type"]))
            if answer is None:
                # A claim the consequence node never saw keeps the detector's
                # answer, like every other unanswered question in this pipeline.
                kept.append(claim)
                continue
            harmless = consequence_module.harmless(answer.get("harm", ""))
            vacuous = bool(answer.get("claims_harm_without_a_run"))
            if harmless:
                counts["named_no_harm"] += 1
            elif vacuous and args.strict_consequence:
                counts["claimed_harm_without_a_run"] += 1
            else:
                kept.append(claim)
                continue
            if not args.quiet:
                print(f"  no harm named  {claim['case_id']}:{claim['line']}  "
                      f"{claim['type']} -- harm={answer.get('harm')} "
                      f"trigger={answer.get('trigger', '')[:60]!r}", file=sys.stderr)
        survivors = kept

    if args.scope_gate:
        kept = []
        for claim in survivors:
            decision = scope_module.resolve([_report(claim)], cases[claim["case_id"]])[0]
            if decision.kept:
                kept.append(claim)
            else:
                counts["out_of_scope"] += 1
                if not args.quiet:
                    print(f"  out-of-scope   {claim['case_id']}:{claim['line']}  "
                          f"{claim['type']} -- {decision.detail}", file=sys.stderr)
        survivors = kept

    if args.evidence_gate:
        kept = []
        for claim in survivors:
            case = cases[claim["case_id"]]
            decision = evidence_module.resolve([_report(claim)], case)[0]
            if decision.kept:
                kept.append(claim)
            else:
                counts["off_operation"] += 1
                if not args.quiet:
                    print(f"  off-operation  {claim['case_id']}:{claim['line']}  "
                          f"{claim['type']} -- {decision.detail}", file=sys.stderr)
        survivors = kept

    if args.one_per_line:
        best: dict[tuple, dict] = {}
        for claim in survivors:
            key = (claim["case_id"], claim["file"], claim["line"])
            if key not in best or claim.get("confidence", 0) > best[key].get("confidence", 0):
                best[key] = claim
        deduped = [c for c in survivors if best[(c["case_id"], c["file"], c["line"])] is c]
        counts["deduped"] = len(survivors) - len(deduped)
        survivors = deduped

    run_dir = args.out / (args.run_id or f"{args.run}-regated")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in survivors),
        encoding="utf-8", newline="\n")
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": run_dir.name, "stage": "regate",
        "regated_run": args.run, "claims_from": claims_file.name, "challenge_run": args.challenge,
        "consequence_run": args.consequence,
        "strict_consequence": bool(args.strict_consequence), "dataset": dataset,
        "scope_gate": bool(args.scope_gate),
        "evidence_gate": bool(args.evidence_gate),
        "cross_file_gate": bool(args.cross_file_gate),
        "one_per_line": bool(args.one_per_line),
        "predictions": len(survivors), **counts,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"claims {counts['claims']}  refuted {counts['refuted']}  "
              f"no-harm {counts['named_no_harm']}  "
              f"out-of-scope {counts['out_of_scope']}  "
              f"off-operation {counts['off_operation']}  "
              f"spared (cross-file) {counts['spared_cross_file']}  "
              f"-> {len(survivors)}")
        print(f"artefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
