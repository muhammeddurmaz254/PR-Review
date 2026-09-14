"""Run the verifier agent over a finished detector run, in place of the challenge.

Reads the gated claims (`adapters.claims_path`), verifies each one in a fresh
context with tools, applies layer 3 to what it cites, and writes the verdicts
plus one prediction file per policy -- `strict` (only established claims) and
`lenient` (drop only contradicted ones) -- so both are scored from one pass.
The chosen policy, `strict` by default, is also written as the run's own
`predictions.jsonl`: this is the default verify stage of the pipeline, in
place of `run_challenge.py`, measured over three corpora in `detect/verify.py`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import claims_path, corpus_of, load_cases
from detect import dedupe, agent, challenge, client as client_module, prompt, verify

HERE = Path(__file__).resolve().parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The per-claim verifier agent.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--contract-evidence", action=argparse.BooleanOptionalAction, default=True,
                        help="count a rule the repository states in prose as evidence (measured: it "
                             "recovers a contract-backed claim on zincir and costs nothing on halka)")
    parser.add_argument("--precedent", action=argparse.BooleanOptionalAction, default=False,
                        help="settle a claim that something is missing by looking for how the "
                             "repository does the same thing elsewhere, including the revision "
                             "before the change; implies --contract-evidence")
    parser.add_argument("--mechanism", action=argparse.BooleanOptionalAction, default=False,
                        help="settle a claim by whether the failing run can be written from the "
                             "code -- the input, the order, the line -- asking nothing of how the "
                             "rest of the repository does it; implies --contract-evidence")
    parser.add_argument("--callers-note", action=argparse.BooleanOptionalAction, default=False,
                        help="with --mechanism: forbid the 'nothing calls it' argument, which "
                             "contradicted three true findings whose callers are a route table, "
                             "a schedule or a later change")
    parser.add_argument("--judge-location", action=argparse.BooleanOptionalAction, default=False,
                        help="show the verifier the claim's own sentence and not its catalogue name or "
                             "definition, so a misfit name cannot decide the verdict (D32); off by "
                             "default -- on its own it lets clean-twin claims through, see run_publish --agree")
    parser.add_argument("--rename", action=argparse.BooleanOptionalAction, default=False,
                        help="judge the line, not the name: an established claim is published under "
                             "the kind the verifier says the evidence shows (detect/verify.py, D31); "
                             "off by default: pooled F1 0.800 -> 0.806 but demo_repo loses one, reproducibly")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True,
                        help="publish one finding per code site: where two claims land within "
                             "--dedupe-radius lines of each other in one file, keep the more "
                             "confident (measured: no true finding lost on any corpus)")
    parser.add_argument("--dedupe-radius", type=int, default=dedupe.RADIUS)
    parser.add_argument("--policy", choices=("strict", "lenient"), default="strict",
                        help="which policy the run's own predictions.jsonl carries; both are always "
                             "written under the run for comparison")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true",
                        help="reuse verdicts already written for this run id; a corpus pass is long "
                             "and a dropped tunnel must not make it start over")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.judge_location and args.rename:
        parser.error("--judge-location and --rename are two answers to one problem; choose one")

    source = args.out / args.run
    manifest = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset, version = manifest["dataset"], manifest["prompt_version"]
    deletions = bool(manifest.get("deletions"))
    definitions = prompt.VERSIONS[version][1][dataset]
    cases = {c.case_id: c for c in load_cases(corpus_of(manifest, HERE / "datasets"))}
    claims_file = claims_path(source)
    claims = [json.loads(l) for l in claims_file.read_text().splitlines() if l.strip()][: args.limit]
    client = client_module.OllamaClient(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
                                        timeout=args.timeout, think=args.think)
    problem = client.health()
    if problem:
        raise SystemExit(f"{args.base_url}: {problem}")
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    key = lambda row: (row["case_id"], row["file"], row["line"], row["type"])
    done: dict[tuple, dict] = {}
    if args.resume and (run_dir / "verdicts.jsonl").exists():
        done = {key(json.loads(l)): json.loads(l)
                for l in (run_dir / "verdicts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    workspaces: dict[str, agent.Workspace] = {}
    verdicts = []
    log = (run_dir / "verdicts.jsonl").open("a" if done else "w", encoding="utf-8", newline="\n")
    for index, claim in enumerate(claims, start=1):
        stored = done.get(key(claim))
        if stored is not None:
            verdicts.append(stored)
            if not args.quiet:
                print(f"[{index}/{len(claims)}] {claim['case_id']}:{claim['line']} reused", file=sys.stderr)
            continue
        case = cases[claim["case_id"]]
        workspace = workspaces.setdefault(case.case_id, agent.Workspace(case))
        rows = challenge.excerpt(case, claim["file"], claim["line"], args.radius, with_deletions=deletions)
        kinds = list(definitions) if args.rename else None
        if args.judge_location:
            user = verify.user_message_located(claim, definitions.get(claim["type"], ""), rows)
        else:
            user = verify.user_message(claim, definitions.get(claim["type"], ""), rows, kinds)
        system = verify.system_for(args.contract_evidence, args.precedent, args.mechanism,
                                   args.callers_note)
        schema, final_ask = verify.SCHEMA, verify.FINAL_ASK
        if args.rename:
            system, schema, final_ask = (verify.with_kind_clause(system), verify.schema_with_kinds(kinds),
                                         verify.FINAL_ASK_KIND)
        text, transcript, calls = agent.review(client.chat, system, user, workspace, schema,
                                               args.max_tool_calls, guide=verify.GUIDE,
                                               final_ask=final_ask)
        answer = verify.parse(text)
        settled = verify.settle(answer, workspace)
        row = {**{k: claim[k] for k in ("case_id", "file", "line", "type")}, "title": claim.get("message", ""),
               "claimed_verdict": answer["verdict"], "verdict": settled, "kind": answer.get("kind", ""),
               "needed": answer.get("needed", ""),
               "evidence": answer.get("evidence", []), "reason": answer.get("reason", ""),
               "evidence_checked": [verify.cited(workspace, e) for e in answer.get("evidence", [])],
               "tool_calls": calls, "steps": transcript,
               "error": next((s.get("error") for s in transcript if s.get("error")), "")}
        verdicts.append(row)
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        if not args.quiet:
            print(f"[{index}/{len(claims)}] {claim['case_id']}:{claim['line']} {claim['type']} -> "
                  f"{answer['verdict']} / {settled} ({calls} calls)", file=sys.stderr, flush=True)
    log.close()

    site_drops = {}
    for policy in ("strict", "lenient"):
        names = list(definitions)
        kept = []
        for c, v in zip(claims, verdicts):
            if not verify.keeps(v["verdict"], policy):
                continue
            kind = verify.published_type(c, v, names) if args.rename else c["type"]
            kept.append({**c, "type": kind, "claimed_type": c["type"]} if kind != c["type"] else c)
        if args.dedupe:
            before = len(kept)
            kept = dedupe.one_per_site(kept, args.dedupe_radius)
            site_drops[policy] = before - len(kept)
        body = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in kept)
        sub = run_dir / policy
        sub.mkdir(exist_ok=True)
        (sub / "predictions.jsonl").write_text(body, encoding="utf-8", newline="\n")
        (sub / "config.json").write_text(json.dumps({**manifest, "run_id": f"{args.run_id}/{policy}",
                                                     "stage": f"verify-agent:{policy}"}, indent=2,
                                                    ensure_ascii=False) + "\n", encoding="utf-8")
        if policy == args.policy:
            # The run's own predictions are the chosen policy's, so `run_eval.py`
            # and anything else downstream reads this stage like any other.
            (run_dir / "predictions.jsonl").write_text(body, encoding="utf-8", newline="\n")
    counts = {v: sum(1 for r in verdicts if r["verdict"] == v) for v in verify.VERDICTS}
    (run_dir / "config.json").write_text(json.dumps({
        **manifest, "run_id": args.run_id, "stage": "verify-agent", "verified_run": args.run,
        "claims_from": claims_file.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": {"model": client.name, "model_build": client.build(), "max_tool_calls": args.max_tool_calls,
                     "num_ctx": args.num_ctx, "contract_evidence": bool(args.contract_evidence),
                     "precedent": bool(args.precedent), "mechanism": bool(args.mechanism),
                     "callers_note": bool(args.callers_note)},
        "claims": len(claims), "reused": len(done), "policy": args.policy, "verdicts": counts,
        "rename": bool(args.rename), "judge_location": bool(args.judge_location),
        "renamed": sum(1 for c, v in zip(claims, verdicts)
                       if args.rename and verify.published_type(c, v, list(definitions)) != c["type"]),
        "dedupe": {"radius": args.dedupe_radius, "dropped": site_drops} if args.dedupe else None,
        "claimed_verdicts": {v: sum(1 for r in verdicts if r["claimed_verdict"] == v) for v in verify.VERDICTS},
        "errors": sum(1 for r in verdicts if r["error"]),
        "python": platform.python_version(), "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"\nverdicts {counts}  artefacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
