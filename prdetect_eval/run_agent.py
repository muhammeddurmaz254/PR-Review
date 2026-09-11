"""Layer 2 as a detector stage: the look-then-answer reviewer over a corpus.

Writes a run directory shaped like `run_detect.py`'s -- predictions before
and after the quote gate, a manifest with the published types and flags --
so `run_continue.py`, `run_challenge.py`, `run_regate.py` and `run_eval.py`
read it unchanged. `transcripts.jsonl` holds every tool call and its result,
because whether the reviewer looked before it claimed is the thing this
stage exists to change, and it has to be read, not inferred from a score.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import eval_path, load_cases
from detect import agent, anchor as anchor_module, client as client_module, contract, pack, prompt
from detect import evidence as evidence_module, scope as scope_module
from run_detect import harness_commit

HERE = Path(__file__).resolve().parent


def _row(case_id: str, report: contract.Report) -> dict:
    return {"case_id": case_id, "file": report.file, "line": report.line, "end_line": report.line,
            "type": report.type, "confidence": report.confidence, "detector": "review.agent",
            "stage": "detect", "message": report.title}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer 2: the tool-using reviewer.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt-version", default="review/v6-broad")
    parser.add_argument("--facts", action="store_true")
    parser.add_argument("--deletions", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-findings", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", action="append", default=[], help="only these case ids (repeatable)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(eval_path(args.dataset, HERE / "datasets"))
    if args.case:
        cases = [c for c in cases if c.case_id in set(args.case)]
    cases = cases[: args.limit]
    version = args.prompt_version
    order = contract.LEGACY_ORDER
    if version in prompt.EVIDENCE_FIRST:
        order = contract.EVIDENCE_ORDER
    elif version in prompt.CLAIM_FIRST:
        order = contract.CLAIM_ORDER
    quoted = version in prompt.QUOTED
    schema = contract.response_schema(prompt.types(args.dataset, version), quote=quoted, order=order)
    client = client_module.OllamaClient(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
                                        timeout=args.timeout, think=args.think)
    problem = client.health()
    if problem:
        raise SystemExit(f"{args.base_url}: {problem}")
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions, anchored, transcripts, failures, tool_calls = [], [], [], 0, 0
    # Written case by case: a corpus run takes hours, its progress has to be
    # readable while it runs, and a dropped tunnel must not lose what is done.
    progress = (run_dir / "transcripts.jsonl").open("w", encoding="utf-8", newline="\n")
    for index, case in enumerate(cases, start=1):
        item = pack.build(case, args.dataset, version, (), args.facts, args.deletions)
        workspace = agent.Workspace(case)
        text, transcript, calls = agent.review(client.chat, item.system, item.user, workspace, schema,
                                               args.max_tool_calls)
        tool_calls += calls
        failures += any(step.get("error") for step in transcript)
        reports, _ = contract.parse(text)
        kept = contract.cap(contract.dedupe(reports), args.max_findings)
        kept = scope_module.apply(scope_module.resolve(kept, case))
        kept = evidence_module.apply(evidence_module.resolve(kept, case))
        predictions += [_row(case.case_id, r) for r in kept]
        if quoted:
            verdicts = anchor_module.resolve(kept, case, with_deletions=args.deletions)
            anchored += [_row(case.case_id, r) for r in anchor_module.apply(verdicts)]
        else:
            anchored += [_row(case.case_id, r) for r in kept]
        transcripts.append({"case_id": case.case_id, "tool_calls": calls, "reported": len(reports),
                            "kept": len(kept), "steps": transcript})
        progress.write(json.dumps(transcripts[-1], ensure_ascii=False) + "\n")
        progress.flush()
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case.case_id}: {calls} tool calls, {len(reports)} raw, "
                  f"{len(kept)} kept", file=sys.stderr, flush=True)

    def write(name: str, rows: list[dict]) -> None:
        (run_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8", newline="\n")
    progress.close()
    write("predictions.jsonl", predictions)
    write("predictions.anchored.jsonl", anchored)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "detect", "detector": "review.agent", "dataset": args.dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(), "harness_commit": harness_commit(),
        "model": client.name, "model_build": client.build(), "prompt_version": version,
        "types": prompt.types(args.dataset, version), "facts": args.facts, "deletions": args.deletions,
        "context": [], "max_findings": args.max_findings, "context_tokens": args.num_ctx,
        "agent": {"max_tool_calls": args.max_tool_calls, "tool_calls": tool_calls},
        "sampling": {"temperature": client.temperature, "seed": client.seed, "think": args.think},
        "cases": len(cases), "predictions": len(predictions), "call_failures": failures,
        "complete": failures == 0, "python": platform.python_version(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"\ntool calls {tool_calls}  predictions {len(predictions)}  artefacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
