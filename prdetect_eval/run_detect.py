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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

from adapters import load_cases, write_predictions
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
    parser.add_argument("--dry-run", action="store_true", help="write prompts and budget only")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="reasoning models only: --no-think keeps the answer parseable")
    parser.add_argument("--limit", type=int, help="first N cases only, for a smoke run")
    parser.add_argument("--case", action="append", default=[], help="restrict to these case ids")
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    eval_path = args.eval or DATASETS / f"{args.dataset}.eval.jsonl"
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

    types = prompt.types(args.dataset)
    schema = contract.response_schema(types)
    packs = [pack.build(case, args.dataset) for case in cases]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "dry-run" if detector is None else detector.name.replace(":", "-").replace("/", "-")
    run_id = args.run_id or f"{stamp}_{args.dataset}_{slug}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "packs.jsonl").write_text("".join(
        json.dumps({
            "case_id": item.case_id, "code_lines": item.shown_lines,
            "estimated_prompt_tokens": item.estimated_tokens, "user": item.user,
        }, ensure_ascii=False) + "\n" for item in packs
    ), encoding="utf-8", newline="\n")
    (run_dir / "system_prompt.txt").write_text(packs[0].system, encoding="utf-8", newline="\n")

    predictions: list[Prediction] = []
    responses: list[client_module.Response] = []
    rejects: list[dict] = []
    failures = 0
    if detector is not None:
        with (run_dir / "responses.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for index, item in enumerate(packs, start=1):
                response = detector.complete(item.system, item.user, schema)
                responses.append(response)
                failures += bool(response.error)
                reports, bad = contract.parse(response.text)
                predictions.extend(to_predictions(item.case_id, reports))
                rejects.extend({
                    "case_id": item.case_id, "stage": reject.stage,
                    "detail": reject.detail, "payload": reject.payload,
                } for reject in bad)
                handle.write(json.dumps({
                    "case_id": item.case_id, "text": response.text, "error": response.error,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "duration_ms": response.duration_ms,
                }, ensure_ascii=False) + "\n")
                # Flushed per call: a corpus run against a rented card takes long
                # enough that the artefact has to be readable while it runs.
                handle.flush()
                if not args.quiet:
                    print(f"[{index}/{len(packs)}] {item.case_id}: "
                          f"{len(reports)} finding(s), {response.duration_ms} ms"
                          + (f"  ERROR {response.error}" if response.error else ""),
                          file=sys.stderr, flush=True)

    write_predictions(run_dir / "predictions.jsonl", predictions)
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
        "quantization": None, "context_tokens": args.num_ctx,
        "prompt_version": prompt.PROMPT_VERSION, "types": types,
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
        if rejects:
            print(f"rejected responses: {len(rejects)}  (see rejects.jsonl)")
        if failures:
            print(f"call failures: {failures}")
        print(f"\nartefacts: {run_dir}")
        if detector is not None:
            print(f"score it:  python run_eval.py --eval {eval_path} "
                  f"--predictions {run_dir / 'predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
