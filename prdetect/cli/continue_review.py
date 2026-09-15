"""Review the changed files a detect run left without a claim.

    python -m prdetect.cli.continue_review --run <detect-run> --model qwen3.8:27b --base-url https://<tunnel>

For every pull request the detector made a claim on, if some changed code files
carry none, the same pack is sent once more with a trailer naming what is already
recorded and what is left (`detect/continuation.py`). New claims go through the
same cap, scope and anchor checks as the detector's own, and land beside them in a
new run directory (`<run>-cont` by default) that later stages read like a detect run.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.cli import runs
from prdetect.detect import anchor, contract, continuation, ollama, pack, prompt, scope


def _row(case_id: str, report: contract.Report) -> dict:
    return {"case_id": case_id, "file": report.file, "line": report.line,
            "end_line": report.line, "type": report.type, "confidence": report.confidence,
            "detector": "review.llm", "stage": "continue", "message": report.title}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review the changed files a detect run left without a claim.")
    parser.add_argument("--run", required=True, help="the detect run")
    parser.add_argument("--model", help="Ollama model tag")
    parser.add_argument("--base-url", default=ollama.DEFAULT_BASE_URL)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", help="write the prompts and stop")
    parser.add_argument("--run-id", help="defaults to <run>-cont")
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = runs.run_dir(args.run, args.out)
    manifest = runs.read_manifest(source)
    limit = int(manifest.get("max_findings") or 0)
    cases = runs.cases_by_id(manifest)
    claims = runs.read_rows(source / "predictions.jsonl")
    anchored_source = runs.read_rows(source / "predictions.anchored.jsonl") or list(claims)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        by_case[claim["case_id"]].append(claim)

    detector = None
    if not args.dry_run:
        if not args.model:
            raise SystemExit("give --model, or --dry-run")
        detector = ollama.OllamaClient(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
                                       temperature=args.temperature, seed=args.seed,
                                       timeout=args.timeout, think=args.think)
        problem = detector.health()
        if problem:
            raise SystemExit(f"{args.base_url}: {problem}")

    run_dir = args.out / (args.run_id or f"{source.name}-cont")
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = contract.response_schema(prompt.types())

    added, added_anchored, trace, responses = [], [], [], []
    counts: dict[str, int] = defaultdict(int)
    jobs = [(case_id, continuation.targets(cases[case_id], {c["file"] for c in by_case[case_id]}))
            for case_id in sorted(by_case)]
    claimed: dict[str, set] = defaultdict(set)
    for claim in claims:
        claimed[claim["case_id"]].add((claim["file"], claim["line"]))
    for case_id, remaining in jobs:
        if not remaining:
            continue
        case = cases[case_id]
        item = pack.build(case, with_facts=True, with_deletions=True)
        reported = by_case[case_id]
        user = item.user + "\n" + continuation.trailer(reported, remaining)
        counts["calls"] += 1
        if detector is None:
            trace.append({"case_id": case_id, "remaining": remaining, "user": user})
            continue
        response = detector.complete(item.system, user, schema)
        responses.append({"case_id": case_id, "files": remaining, "text": response.text,
                          "error": response.error, "prompt_tokens": response.prompt_tokens,
                          "completion_tokens": response.completion_tokens,
                          "duration_ms": response.duration_ms})
        reports, _ = contract.parse(response.text)
        counts["raw"] += len(reports)
        inside = continuation.within(reports, remaining)
        counts["outside_remaining"] += len(reports) - len(inside)
        new_ones = continuation.fresh(contract.dedupe(inside), claimed[case_id])
        counts["already_claimed"] += len(inside) - len(new_ones)
        # The pull request keeps the detect run's cap across both calls.
        room = max(0, limit - len(reported) - sum(1 for a in added if a["case_id"] == case_id))
        kept = contract.cap(new_ones, room)
        counts["dropped_over_cap"] += len(new_ones) - len(kept)
        claimed[case_id].update((r.file, r.line) for r in kept)
        placements = scope.resolve(kept, case)
        counts["out_of_scope"] += sum(1 for p in placements if not p.kept)
        kept = scope.apply(placements)
        added.extend(_row(case_id, r) for r in kept)
        decisions = anchor.resolve(kept, case, with_deletions=True)
        added_anchored.extend(_row(case_id, r) for r in anchor.apply(decisions))
        trace.append({"case_id": case_id, "remaining": remaining,
                      "reported": [(r.file, r.line, r.type) for r in reports],
                      "kept": [(r.file, r.line, r.type) for r in kept]})
        if not args.quiet:
            print(f"[{counts['calls']}] {case_id} {remaining}: {len(reports)} raw, {len(kept)} kept"
                  + (f"  ERROR {response.error}" if response.error else ""), file=sys.stderr, flush=True)

    runs.write_rows(run_dir / "predictions.jsonl", claims + added)
    runs.write_rows(run_dir / "predictions.anchored.jsonl", anchored_source + added_anchored)
    runs.write_rows(run_dir / "continuation.jsonl", trace)
    if responses:
        runs.write_rows(run_dir / "responses.jsonl", responses)
    runs.write_manifest(run_dir, {
        **manifest, "run_id": run_dir.name, "stage": "continue", "continued_run": source.name,
        "continuation": dict(counts), "added": len(added),
        "complete": detector is not None and not any(r["error"] for r in responses),
        **runs.provenance(argv),
    })
    if not args.quiet:
        print(f"\ncalls {counts['calls']}  added {len(added)}  {dict(counts)}\n-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
