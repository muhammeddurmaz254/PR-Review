"""Check every claim of a detect or continue run with the verifier.

    python -m prdetect.cli.verify --run <continue-run> --run-id <id>-named   --model qwen3.8:27b --base-url https://<tunnel>
    python -m prdetect.cli.verify --run <continue-run> --run-id <id>-located --judge-location --model ...

Each claim is verified in a fresh context with tools (`detect/verify.py`), and
what the verifier cites is checked against the repository. The pipeline runs this
twice over the same claims -- once with the catalogue name, once with only the
claim's own sentence -- and `publish --agree` keeps what both established.

Writes `runs/<run-id>/`: `verdicts.jsonl`, one row per claim with the evidence
and the tool calls; `predictions.jsonl`, the established claims, one per site;
and `config.json`. `--resume` reuses the verdicts already written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.cli import runs
from prdetect.detect import agent, dedupe, ollama, pack, prompt, verify


def _key(row: dict) -> tuple:
    return row["case_id"], row["file"], row["line"], row["type"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check every claim of a run with the verifier.")
    parser.add_argument("--run", required=True, help="the detect or continue run whose claims are checked")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=ollama.DEFAULT_BASE_URL)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--radius", type=int, default=12, help="lines shown on each side of the claim")
    parser.add_argument("--judge-location", action="store_true",
                        help="show the claim's own sentence instead of its catalogue name and definition")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="reuse the verdicts already written for this run id")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = runs.run_dir(args.run, args.out)
    manifest = runs.read_manifest(source)
    definitions = prompt.definitions()
    cases = runs.cases_by_id(manifest)
    claims_path = runs.claims_file(source)
    claims = runs.read_rows(claims_path)[: args.limit]
    client = ollama.OllamaClient(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
                                 timeout=args.timeout, think=args.think)
    problem = client.health()
    if problem:
        raise SystemExit(f"{args.base_url}: {problem}")
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    done: dict[tuple, dict] = {}
    if args.resume:
        done = {_key(row): row for row in runs.read_rows(run_dir / "verdicts.jsonl")}
    workspaces: dict[str, agent.Workspace] = {}
    verdicts = []
    with (run_dir / "verdicts.jsonl").open("a" if done else "w", encoding="utf-8", newline="\n") as log:
        for index, claim in enumerate(claims, start=1):
            stored = done.get(_key(claim))
            if stored is not None:
                verdicts.append(stored)
                if not args.quiet:
                    print(f"[{index}/{len(claims)}] {claim['case_id']}:{claim['line']} reused", file=sys.stderr)
                continue
            case = cases[claim["case_id"]]
            workspace = workspaces.setdefault(case.case_id, agent.Workspace(case))
            rows = pack.excerpt(case, claim["file"], claim["line"], args.radius, with_deletions=True)
            question = verify.user_message_located if args.judge_location else verify.user_message
            user = question(claim, definitions.get(claim["type"], ""), rows)
            text, transcript, calls = agent.review(client.chat, verify.SYSTEM, user, workspace, verify.SCHEMA,
                                                   args.max_tool_calls, guide=verify.GUIDE,
                                                   final_ask=verify.FINAL_ASK)
            answer = verify.parse(text)
            settled = verify.settle(answer, workspace)
            row = {**{k: claim[k] for k in ("case_id", "file", "line", "type")}, "title": claim.get("message", ""),
                   "claimed_verdict": answer["verdict"], "verdict": settled,
                   "needed": answer.get("needed", ""), "evidence": answer.get("evidence", []),
                   "reason": answer.get("reason", ""),
                   "evidence_checked": [verify.cited(workspace, e) for e in answer.get("evidence", [])],
                   "tool_calls": calls, "steps": transcript,
                   "error": next((s.get("error") for s in transcript if s.get("error")), "")}
            verdicts.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if not args.quiet:
                print(f"[{index}/{len(claims)}] {claim['case_id']}:{claim['line']} {claim['type']} -> "
                      f"{answer['verdict']} / {settled} ({calls} calls)", file=sys.stderr, flush=True)

    established = [claim for claim, row in zip(claims, verdicts) if row["verdict"] == "established"]
    kept = dedupe.one_per_site(established)
    runs.write_rows(run_dir / "predictions.jsonl", kept)
    counts = {v: sum(1 for r in verdicts if r["verdict"] == v) for v in verify.VERDICTS}
    runs.write_manifest(run_dir, {
        **manifest, "run_id": args.run_id, "stage": "verify", "verified_run": source.name,
        "claims_from": claims_path.name, "question": "located" if args.judge_location else "named",
        "verifier": {"model": client.name, "model_build": client.build(),
                     "max_tool_calls": args.max_tool_calls, "num_ctx": args.num_ctx, "radius": args.radius},
        "claims": len(claims), "reused": len(done), "verdicts": counts,
        "claimed_verdicts": {v: sum(1 for r in verdicts if r["claimed_verdict"] == v) for v in verify.VERDICTS},
        "published": len(kept), "errors": sum(1 for r in verdicts if r["error"]),
        **runs.provenance(argv),
    })
    if not args.quiet:
        print(f"\nverdicts {counts}  -> {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
