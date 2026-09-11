"""Run stage [4b] over a finished detector run.

For every pull request the detector reported on, if some changed code files
carry no report, ask once more about those files alone -- the whole pack, the
same pinned system prompt, and a trailer naming what is already recorded and
what is left. See ``detect/continuation.py`` for why, and for the exposure that
was measured before this was written.

The new reports go through the gates the detector's own went through -- cap,
scope, evidence, and the quote gate -- and land beside the source run's
predictions in a new run directory. That directory carries the source manifest,
so ``run_challenge.py``, ``run_regate.py`` and ``run_eval.py`` read it exactly
like a detector run: the published types, the deletions flag and the dataset
come through unchanged.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import eval_path, load_cases
from detect import anchor as anchor_module
from detect import client as client_module
from detect import contract, continuation, pack, prompt
from detect import evidence as evidence_module
from detect import scope as scope_module
from run_detect import harness_commit

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"


def _row(case_id: str, report: contract.Report) -> dict:
    return {"case_id": case_id, "file": report.file, "line": report.line,
            "end_line": report.line, "type": report.type, "confidence": report.confidence,
            "detector": "review.llm", "stage": "continue", "message": report.title}


def _schema(version: str, dataset: str) -> dict:
    order = contract.LEGACY_ORDER
    if version in prompt.OPEN:
        order = contract.OPEN_ORDER
    elif version in prompt.EVIDENCE_FIRST:
        order = contract.EVIDENCE_ORDER
    elif version in prompt.CLAIM_FIRST:
        order = contract.CLAIM_ORDER
    return contract.response_schema(prompt.types(dataset, version),
                                    quote=version in prompt.QUOTED, order=order)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review the changed files a detector run skipped.")
    parser.add_argument("--run", required=True, help="detector run id under runs/")
    parser.add_argument("--model", help="Ollama model tag")
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="write the prompts and stop")
    parser.add_argument("--run-id", help="defaults to <run>-cont")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = args.out / args.run
    manifest = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset, version = manifest["dataset"], manifest["prompt_version"]
    facts, deletions = bool(manifest.get("facts")), bool(manifest.get("deletions"))
    context = tuple(manifest.get("context") or ())
    limit = int(manifest.get("max_findings") or 3)
    quoted = version in prompt.QUOTED
    cases = {c.case_id: c for c in load_cases(eval_path(dataset, DATASETS))}
    claims = [json.loads(l) for l in (source / "predictions.jsonl").read_text().splitlines() if l.strip()]
    anchored_path = source / "predictions.anchored.jsonl"
    anchored_src = ([json.loads(l) for l in anchored_path.read_text().splitlines() if l.strip()]
                    if anchored_path.exists() else list(claims))
    by_case: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        by_case[claim["case_id"]].append(claim)

    detector = None
    if not args.dry_run:
        if not args.model:
            raise SystemExit("give --model, or --dry-run")
        detector = client_module.OllamaClient(
            model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
            temperature=args.temperature, seed=args.seed, timeout=args.timeout, think=args.think)
        problem = detector.health()
        if problem:
            raise SystemExit(f"{args.base_url}: {problem}")

    run_dir = args.out / (args.run_id or f"{args.run}-cont")
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = _schema(version, dataset)

    added, added_anchored, trace, responses = [], [], [], []
    counts = defaultdict(int)
    for case_id in sorted(by_case):
        case = cases[case_id]
        remaining = continuation.targets(case, {c["file"] for c in by_case[case_id]})
        if not remaining:
            continue
        item = pack.build(case, dataset, version, context, facts, deletions)
        user = item.user + "\n" + continuation.trailer(by_case[case_id], remaining)
        counts["calls"] += 1
        if detector is None:
            trace.append({"case_id": case_id, "remaining": remaining, "user": user})
            continue
        response = detector.complete(item.system, user, schema)
        responses.append({"case_id": case_id, "text": response.text, "error": response.error,
                          "prompt_tokens": response.prompt_tokens,
                          "completion_tokens": response.completion_tokens,
                          "duration_ms": response.duration_ms})
        reports, _ = contract.parse(response.text)
        counts["raw"] += len(reports)
        inside = continuation.within(reports, remaining)
        counts["outside_remaining"] += len(reports) - len(inside)
        room = max(0, limit - len(by_case[case_id]))
        kept = contract.cap(contract.dedupe(inside), room)
        counts["dropped_over_cap"] += len(inside) - len(kept)
        placements = scope_module.resolve(kept, case)
        counts["out_of_scope"] += sum(1 for p in placements if not p.kept)
        kept = scope_module.apply(placements)
        calls = evidence_module.resolve(kept, case)
        counts["off_operation"] += sum(1 for c in calls if not c.kept)
        kept = evidence_module.apply(calls)
        added.extend(_row(case_id, r) for r in kept)
        if quoted:
            verdicts = anchor_module.resolve(kept, case, context=context, with_deletions=deletions)
            added_anchored.extend(_row(case_id, r) for r in anchor_module.apply(verdicts))
        else:
            added_anchored.extend(_row(case_id, r) for r in kept)
        trace.append({"case_id": case_id, "remaining": remaining,
                      "reported": [(r.file, r.line, r.type) for r in reports],
                      "kept": [(r.file, r.line, r.type) for r in kept]})
        if not args.quiet:
            print(f"[{counts['calls']}] {case_id}: {len(reports)} raw, {len(kept)} kept"
                  + (f"  ERROR {response.error}" if response.error else ""), file=sys.stderr, flush=True)

    def write(name: str, rows: list[dict]) -> None:
        (run_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8", newline="\n")

    write("predictions.jsonl", claims + added)
    write("predictions.anchored.jsonl", anchored_src + added_anchored)
    write("continuation.jsonl", trace)
    if responses:
        write("responses.jsonl", responses)
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": run_dir.name, "stage": "continue",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "continued_run": args.run, "harness_commit": harness_commit(),
        "continuation": dict(counts), "added": len(added),
        "complete": detector is not None and not any(r["error"] for r in responses),
        "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"\ncalls {counts['calls']}  added {len(added)}  {dict(counts)}\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
