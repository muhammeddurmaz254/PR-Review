"""Run the review detector over one dataset.

Enumerate nothing, pack the pull request, ask once, parse, write. Scoring is
deliberately elsewhere: the output is a ``predictions.jsonl`` that
``run_eval.py`` reads exactly as it reads a baseline, so changing a metric never
costs a second pass over a model.

Three modes, in the order they should be used:

``--dry-run``      no server. Writes every prompt and reports the context budget.
``--stub silent``  no server. Reports nothing: the precision floor.
``--model NAME``   the real thing, against Ollama at ``--base-url``.
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

from adapters import load_cases, write_predictions
from detect import anchor as anchor_module
from detect import scope as scope_module
from detect import evidence as evidence_module
from detect import client as client_module
from detect import contract, pack, prompt
from schema import Case, Prediction, Span

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATASETS = HERE / "datasets"
DETECTOR = "review.llm"


def harness_commit() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"), cwd=HERE, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def to_predictions(case_id: str, reports: Sequence[contract.Report]) -> list[Prediction]:
    return [
        Prediction(
            case_id=case_id, span=Span(report.file, report.line, report.line),
            type=report.type, confidence=report.confidence,
            detector=DETECTOR, stage="detect", message=report.title,
        )
        for report in reports
    ]


def budget(packs: Sequence[pack.Pack], responses: Sequence[client_module.Response]) -> dict:
    estimated = [item.estimated_tokens for item in packs]
    shown = [item.shown_lines for item in packs]
    measured = [r.prompt_tokens for r in responses if r.prompt_tokens]
    summary = {
        "packs": len(packs),
        "estimated_prompt_tokens_mean": round(sum(estimated) / len(estimated), 1) if estimated else 0,
        "estimated_prompt_tokens_max": max(estimated, default=0),
        "code_lines_mean": round(sum(shown) / len(shown), 1) if shown else 0,
        "code_lines_max": max(shown, default=0),
    }
    if measured:
        summary |= {
            "measured_prompt_tokens_mean": round(sum(measured) / len(measured), 1),
            "measured_prompt_tokens_median": median(measured),
            "measured_prompt_tokens_max": max(measured),
        }
    latency = [r.duration_ms for r in responses if r.duration_ms]
    if latency:
        summary |= {"latency_ms_mean": round(sum(latency) / len(latency)), "latency_ms_max": max(latency)}
    return summary


def resolve_client(args: argparse.Namespace) -> client_module.Client | None:
    if args.dry_run:
        return None
    if args.stub:
        return client_module.STUBS[args.stub]()
    if not args.model:
        raise SystemExit("give --model, or --stub NAME, or --dry-run")
    return client_module.OllamaClient(
        model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
        temperature=args.temperature, seed=args.seed, timeout=args.timeout, think=args.think,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the review detector over one dataset.")
    parser.add_argument("--dataset", default="demo_repo", choices=sorted(prompt.TAXONOMIES))
    parser.add_argument("--eval", type=Path, help="defaults to datasets/<dataset>.eval.jsonl")
    parser.add_argument("--model", help="Ollama model tag, e.g. qwen3.8:27b")
    parser.add_argument("--base-url", default=client_module.DEFAULT_BASE_URL,
                        help="Ollama server; an ngrok https URL when the card is remote")
    parser.add_argument("--stub", choices=sorted(client_module.STUBS), help="run without a server")
    parser.add_argument("--prompt-version", default=prompt.PROMPT_VERSION,
                        choices=sorted(prompt.VERSIONS),
                        help="v1 decided the precision/recall trade-off inside the model; "
                             "v2 grades the confidence and leaves it to --threshold")
    parser.add_argument("--dry-run", action="store_true", help="write prompts and budget only")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="reasoning models only: --no-think keeps the answer parseable")
    # Off by default: measured, it cut the raw findings from sixty-five to
    # thirty and moved no metric, because the per-pull-request cap was already
    # discarding exactly that surplus at no extra call. Kept for the case it
    # does serve -- bounding the prompt when the context window is small.
    parser.add_argument("--resume", action="store_true",
                        help="reuse the answers already in the run directory and ask only "
                             "for the packs still missing")
    parser.add_argument("--facts", action="store_true",
                        help="state what the repository says about the names this change "
                             "defines; silent unless it has something discriminating to say")
    parser.add_argument("--deletions", action="store_true",
                        help="print the lines this pull request deleted, where they were; "
                             "lines that come back elsewhere in the file are not removals")
    parser.add_argument("--context", action="append", default=[], metavar="GLOB",
                        help="carry unchanged files matching this pattern (repeatable). "
                             "Everything ('*') was measured and lost; narrow wins or nothing does")
    parser.add_argument("--with-repo", action="store_true",
                        help="shorthand for --context '*'")
    parser.add_argument("--max-pack-lines", type=int, default=0,
                        help="split a pull request larger than this into excerpts; 0 never splits")
    parser.add_argument("--scope-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="drop a report about a file this pull request does not change, "
                             "and every report on a pull request that shows no code (stage [5])")
    parser.add_argument("--evidence-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="drop a finding whose type names an operation the accused "
                             "statement does not perform (stage [5b])")
    parser.add_argument("--max-findings", type=int, default=3,
                        help="most confident N per pull request; 0 keeps them all")
    parser.add_argument("--limit", type=int, help="first N cases only, for a smoke run")
    parser.add_argument("--case", action="append", default=[], help="restrict to these case ids")
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # zincir_bench ships two files, and only one of them is open. A run that
    # names the dataset and not the file gets `dev`; reaching `holdout` has to
    # be typed out, counted, and written down in the corpus README (B9).
    default_eval = DATASETS / ("zincir_dev.eval.jsonl" if args.dataset == "zincir"
                               else f"{args.dataset}.eval.jsonl")
    eval_path = args.eval or default_eval
    cases: list[Case] = load_cases(eval_path)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")

    detector = resolve_client(args)
    if isinstance(detector, client_module.OllamaClient):
        problem = detector.health()
        if problem:
            raise SystemExit(f"{args.base_url}: {problem}")

    types = prompt.types(args.dataset, args.prompt_version)
    quoted = args.prompt_version in prompt.QUOTED
    order = contract.LEGACY_ORDER
    if args.prompt_version in prompt.OPEN:
        order = contract.OPEN_ORDER
    elif args.prompt_version in prompt.EVIDENCE_FIRST:
        order = contract.EVIDENCE_ORDER
    elif args.prompt_version in prompt.CLAIM_FIRST:
        order = contract.CLAIM_ORDER
    schema = contract.response_schema(types, quote=quoted, order=order)
    context = tuple(args.context) or (("*",) if args.with_repo else ())
    packs = [item for case in cases
             for item in pack.split(case, args.dataset, args.prompt_version,
                                    args.max_pack_lines, context, args.facts,
                                    args.deletions)]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "dry-run" if detector is None else detector.name.replace(":", "-").replace("/", "-")
    version_slug = args.prompt_version.replace("/", "-")
    run_id = args.run_id or f"{stamp}_{args.dataset}_{slug}_{version_slug}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # A corpus run against a rented card is long enough that the tunnel outlives
    # it only sometimes: rung 1 on halka_bench is 110 calls at two minutes each,
    # and it has been lost twice at thirteen. Answers already on disk are reused
    # rather than paid for again -- but only when the prompt behind them is the
    # same text, because a run stitched from two prompts is worse than no run.
    build = detector.build() if isinstance(detector, client_module.OllamaClient) else ""
    done: dict[tuple[str, int], dict] = {}
    if args.resume and (run_dir / "responses.jsonl").exists():
        stored = run_dir / "system_prompt.txt"
        if stored.exists() and stored.read_text(encoding="utf-8") != packs[0].system:
            raise SystemExit(
                f"{run_dir} was written under a different system prompt; "
                f"resume would mix two. Use a new --run-id."
            )
        if (run_dir / "config.json").exists():
            was = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("model_build")
            if was and build and was != build:
                raise SystemExit(
                    f"{run_dir} was answered by model build {was}, this server serves {build}; "
                    f"resume would mix two. Use a new --run-id."
                )
        for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["case_id"], row.get("part", 1))] = row

    (run_dir / "packs.jsonl").write_text("".join(
        json.dumps({
            "case_id": item.case_id, "part": item.part, "parts": item.parts,
            "code_lines": item.shown_lines,
            "estimated_prompt_tokens": item.estimated_tokens, "user": item.user,
        }, ensure_ascii=False) + "\n" for item in packs
    ), encoding="utf-8", newline="\n")
    (run_dir / "system_prompt.txt").write_text(packs[0].system, encoding="utf-8", newline="\n")

    predictions: list[Prediction] = []
    anchored: list[Prediction] = []
    decisions: list[dict] = []
    signatures: list[dict] = []
    responses: list[client_module.Response] = []
    rejects: list[dict] = []
    failures = 0
    off_operation = 0
    out_of_scope = 0
    dropped = 0
    by_case = {case.case_id: case for case in cases}
    # Reports are gathered per pull request and capped once, not per excerpt:
    # the cap says how many comments one review may leave, and splitting a large
    # pull request must not multiply that.
    harvest: dict[str, list[contract.Report]] = {case.case_id: [] for case in cases}
    if detector is not None:
        mode = "a" if done else "w"
        with (run_dir / "responses.jsonl").open(mode, encoding="utf-8", newline="\n") as handle:
            for index, item in enumerate(packs, start=1):
                stored = done.get((item.case_id, item.part))
                if stored is not None and not stored.get("error"):
                    reports, bad = contract.parse(stored["text"])
                    harvest[item.case_id].extend(reports)
                    responses.append(client_module.Response(
                        text=stored["text"], thinking=stored.get("thinking", ""),
                        prompt_tokens=int(stored.get("prompt_tokens", 0)),
                        completion_tokens=int(stored.get("completion_tokens", 0)),
                        duration_ms=int(stored.get("duration_ms", 0)),
                    ))
                    continue
                response = detector.complete(item.system, item.user, schema)
                responses.append(response)
                failures += bool(response.error)
                reports, bad = contract.parse(response.text)
                harvest[item.case_id].extend(reports)
                rejects.extend({
                    "case_id": item.case_id, "stage": reject.stage,
                    "detail": reject.detail, "payload": reject.payload,
                } for reject in bad)
                handle.write(json.dumps({
                    "case_id": item.case_id, "part": item.part, "parts": item.parts,
                    "text": response.text, "thinking": response.thinking,
                    "error": response.error,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "duration_ms": response.duration_ms,
                }, ensure_ascii=False) + "\n")
                # Flushed per call: a corpus run against a rented card takes long
                # enough that the artefact has to be readable while it runs.
                handle.flush()
                if not args.quiet:
                    where = f" [{item.part}/{item.parts}]" if item.parts > 1 else ""
                    print(f"[{index}/{len(packs)}] {item.case_id}{where}: "
                          f"{len(reports)} finding(s), {response.duration_ms} ms"
                          + (f"  ERROR {response.error}" if response.error else ""),
                          file=sys.stderr, flush=True)

        for case_id, found in harvest.items():
            kept = contract.cap(contract.dedupe(found), args.max_findings)
            dropped += len(found) - len(kept)
            if args.scope_gate:
                # Stage [5] proper: a report the pull request cannot be about is
                # refused before anything reads the line it names.
                placements = scope_module.resolve(kept, by_case[case_id])
                out_of_scope += sum(1 for p in placements if not p.kept)
                signatures.extend({
                    "case_id": case_id, "file": p.report.file, "line": p.report.line,
                    "type": p.report.type, "verdict": "out_of_scope" if not p.kept else "in_scope",
                    "detail": p.detail, "title": p.report.title,
                } for p in placements if not p.kept)
                kept = scope_module.apply(placements)
            if args.evidence_gate:
                # Stage [5b] runs before the quote gate: both refuse a report for
                # pointing at the wrong place, and neither reads a label, so the
                # order only decides which reason is recorded first.
                calls = evidence_module.resolve(kept, by_case[case_id])
                off_operation += sum(1 for call in calls if not call.kept)
                signatures.extend({
                    "case_id": case_id, "file": call.report.file, "line": call.report.line,
                    "type": call.report.type, "verdict": call.verdict,
                    "detail": call.detail, "title": call.report.title,
                } for call in calls)
                kept = evidence_module.apply(calls)
            predictions.extend(to_predictions(case_id, kept))
            if quoted:
                verdicts = anchor_module.resolve(kept, by_case[case_id], context=context,
                                                 with_deletions=args.deletions)
                anchored.extend(to_predictions(case_id, anchor_module.apply(verdicts)))
                decisions.extend({
                    "case_id": case_id, "file": v.report.file,
                    "claimed_line": v.report.line, "line": v.line,
                    "verdict": v.verdict, "detail": v.detail,
                    "quote": v.report.quote, "title": v.report.title,
                } for v in verdicts)

    write_predictions(run_dir / "predictions.jsonl", predictions)
    if args.evidence_gate:
        (run_dir / "signatures.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in signatures),
            encoding="utf-8", newline="\n")
    if quoted:
        # Both files are kept so the prompt change and the filter can be scored
        # apart: one asks whether requesting a quote moved the model, the other
        # whether checking it is worth anything.
        write_predictions(run_dir / "predictions.anchored.jsonl", anchored)
        (run_dir / "anchors.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
            encoding="utf-8", newline="\n")
    (run_dir / "rejects.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejects),
        encoding="utf-8", newline="\n",
    )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "corpus": str(eval_path), "cases": len(cases),
        "scored_labels": sum(len(case.scored_labels) for case in cases),
        "predictions": len(predictions), "rejected": len(rejects), "call_failures": failures,
        "model": None if detector is None else detector.name,
        "model_build": build or None,
        "quantization": None, "context_tokens": args.num_ctx,
        "prompt_version": args.prompt_version, "types": types,
        "max_findings": args.max_findings, "dropped_over_cap": dropped,
        "scope_gate": bool(args.scope_gate),
        "dropped_out_of_scope": out_of_scope,
        "evidence_gate": bool(args.evidence_gate),
        "dropped_off_operation": off_operation,
        "field_order": list(order),
        "max_pack_lines": args.max_pack_lines, "packs": len(packs),
        "context": list(context), "facts": args.facts, "deletions": args.deletions,
        "reused_answers": len(done),
        # A run that lost its server two thirds of the way through still writes
        # every artefact, because that is what --resume reads. It must not also
        # look finished: the tunnel died at call 28 of 110 once and the run
        # exited 0 with a predictions file that would have scored as rung 1.
        "complete": detector is None or failures == 0,
        "anchors": (Counter(row["verdict"] for row in decisions) if quoted else None),
        "detectors": [DETECTOR],
        "sampling": {"temperature": args.temperature, "seed": args.seed, "think": args.think},
        "base_url": args.base_url if isinstance(detector, client_module.OllamaClient) else None,
        "budget": budget(packs, responses),
        "harness_commit": harness_commit(),
        "python": platform.python_version(), "platform": platform.platform(),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "notes": args.note,
    }
    (run_dir / "config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    if not args.quiet:
        cost = manifest["budget"]
        print(f"dataset {args.dataset}  cases {len(cases)}  predictions {len(predictions)}")
        print(f"prompt tokens: estimated mean {cost['estimated_prompt_tokens_mean']} "
              f"max {cost['estimated_prompt_tokens_max']}"
              + (f" | measured mean {cost['measured_prompt_tokens_mean']} "
                 f"max {cost['measured_prompt_tokens_max']}"
                 if "measured_prompt_tokens_mean" in cost else ""))
        if done:
            print(f"reused {len(done)} answer(s) already on disk")
        if dropped:
            print(f"dropped over the {args.max_findings}-per-PR cap: {dropped}")
        if args.scope_gate and out_of_scope:
            print(f"dropped for naming something this pull request does not change: {out_of_scope}")
        if args.evidence_gate and off_operation:
            print(f"dropped for naming an operation the statement does not perform: {off_operation}")
        if quoted:
            counts = Counter(row["verdict"] for row in decisions)
            print("anchors: " + "  ".join(f"{name}={n}" for name, n in counts.most_common())
                  + f"  -> {len(anchored)} of {len(predictions)} kept")
        if rejects:
            print(f"rejected responses: {len(rejects)}  (see rejects.jsonl)")
        print(f"\nartefacts: {run_dir}")
    if failures:
        print(f"\n!! {failures} of {len(packs)} calls failed; this run is INCOMPLETE.",
              file=sys.stderr)
        print(f"!! Do not score it. Re-run the same command with --resume once the "
              f"server is back; only the failed calls are asked again.", file=sys.stderr)
        return 1
    if not args.quiet and detector is not None:
        print(f"score it:  python run_eval.py --eval {eval_path} "
              f"--predictions {run_dir / 'predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
