"""Ask the model about every pull request of a repository.

    python -m prdetect.cli.detect --repo genis_olcum_reposu --model qwen3.8:27b --base-url https://<tunnel>
    python -m prdetect.cli.detect --repo genis_olcum_reposu --dry-run

One call per pull request, with the changed files, the lines it deleted and the
counted facts (`detect/pack.py`). The claims pass three checks before they are
written: at most `--max-findings` per pull request, the most confident first; only
files the pull request changed (`detect/scope.py`); and the quoted line must be
one the model was shown (`detect/anchor.py`), which also corrects a line number
the quote contradicts.

Writes `runs/<run-id>/`: `packs.jsonl`, `system_prompt.txt`, `responses.jsonl`,
`predictions.jsonl` (after the cap and scope), `predictions.anchored.jsonl` (after
the anchor check; what the later stages read), `out_of_scope.jsonl`,
`anchors.jsonl`, `rejects.jsonl` and `config.json`. A run whose calls partly failed
exits 1 and is marked incomplete; the same command with `--resume` asks only for
what is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

from prdetect import paths
from prdetect.cases import Case, Prediction, Span, load_cases, prediction_row
from prdetect.cli import runs
from prdetect.detect import anchor, contract, ollama, pack, prompt, scope

DETECTOR = "review.llm"


def to_predictions(case_id: str, reports: Sequence[contract.Report]) -> list[Prediction]:
    return [Prediction(case_id=case_id, span=Span(report.file, report.line, report.line),
                       type=report.type, confidence=report.confidence,
                       detector=DETECTOR, stage="detect", message=report.title)
            for report in reports]


def budget(packs: Sequence[pack.Pack], responses: Sequence[ollama.Response]) -> dict:
    estimated = [item.estimated_tokens for item in packs]
    shown = [item.shown_lines for item in packs]
    summary = {
        "packs": len(packs),
        "estimated_prompt_tokens_mean": round(sum(estimated) / len(estimated), 1) if estimated else 0,
        "estimated_prompt_tokens_max": max(estimated, default=0),
        "code_lines_mean": round(sum(shown) / len(shown), 1) if shown else 0,
        "code_lines_max": max(shown, default=0),
    }
    measured = [r.prompt_tokens for r in responses if r.prompt_tokens]
    if measured:
        summary |= {"measured_prompt_tokens_mean": round(sum(measured) / len(measured), 1),
                    "measured_prompt_tokens_median": median(measured),
                    "measured_prompt_tokens_max": max(measured)}
    latency = [r.duration_ms for r in responses if r.duration_ms]
    if latency:
        summary |= {"latency_ms_mean": round(sum(latency) / len(latency)), "latency_ms_max": max(latency)}
    return summary


def resolve_client(args: argparse.Namespace) -> ollama.Client | None:
    if args.dry_run:
        return None
    if args.stub:
        return ollama.STUBS[args.stub]()
    if not args.model:
        raise SystemExit("give --model, or --stub NAME, or --dry-run")
    return ollama.OllamaClient(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
                               temperature=args.temperature, seed=args.seed, timeout=args.timeout,
                               think=args.think)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the model about every pull request of a repository.")
    parser.add_argument("--repo", required=True, help="repository slug; its cases are data/cases/<repo>.jsonl")
    parser.add_argument("--cases", type=Path, help="a cases file other than data/cases/<repo>.jsonl")
    parser.add_argument("--model", help="Ollama model tag, e.g. qwen3.8:27b")
    parser.add_argument("--base-url", default=ollama.DEFAULT_BASE_URL,
                        help="Ollama server; an ngrok https URL when the GPU is remote")
    parser.add_argument("--stub", choices=sorted(ollama.STUBS), help="run without a server")
    parser.add_argument("--dry-run", action="store_true", help="write the prompts and stop")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-findings", type=int, default=6,
                        help="most confident N per pull request; 0 keeps them all")
    parser.add_argument("--resume", action="store_true",
                        help="reuse the answers already in the run directory and ask only for the rest")
    parser.add_argument("--limit", type=int, help="first N pull requests only")
    parser.add_argument("--case", action="append", default=[], help="only these cases, e.g. PR-7 (repeatable)")
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases_path = args.cases or paths.cases_file(args.repo)
    cases: list[Case] = load_cases(cases_path)
    if args.case:
        cases = [case for case in cases if case.case_id in set(args.case)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")

    detector = resolve_client(args)
    if isinstance(detector, ollama.OllamaClient):
        problem = detector.health()
        if problem:
            raise SystemExit(f"{args.base_url}: {problem}")

    schema = contract.response_schema(prompt.types())
    packs = [pack.build(case, with_facts=True, with_deletions=True) for case in cases]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "dry-run" if detector is None else detector.name.replace(":", "-").replace("/", "-")
    run_id = args.run_id or f"{stamp}_{args.repo}_{slug}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Answers already on disk are reused rather than paid for again -- but only
    # when the prompt and the model build behind them are the same, because a run
    # stitched from two prompts or two models is not one run.
    build = detector.build() if isinstance(detector, ollama.OllamaClient) else ""
    done: dict[str, dict] = {}
    if args.resume and (run_dir / "responses.jsonl").exists():
        stored = run_dir / "system_prompt.txt"
        if stored.exists() and stored.read_text(encoding="utf-8") != packs[0].system:
            raise SystemExit(f"{run_dir} was written under a different system prompt; "
                             f"resume would mix two. Use a new --run-id.")
        if (run_dir / "config.json").exists():
            was = runs.read_manifest(run_dir).get("model_build")
            if was and build and was != build:
                raise SystemExit(f"{run_dir} was answered by model build {was}, this server serves {build}; "
                                 f"resume would mix two. Use a new --run-id.")
        done = {row["case_id"]: row for row in runs.read_rows(run_dir / "responses.jsonl")}

    runs.write_rows(run_dir / "packs.jsonl", ({
        "case_id": item.case_id, "code_lines": item.shown_lines,
        "estimated_prompt_tokens": item.estimated_tokens, "user": item.user,
    } for item in packs))
    (run_dir / "system_prompt.txt").write_text(packs[0].system, encoding="utf-8", newline="\n")

    predictions: list[Prediction] = []
    anchored: list[Prediction] = []
    anchors: list[dict] = []
    out_of_scope: list[dict] = []
    responses: list[ollama.Response] = []
    rejects: list[dict] = []
    failures = dropped = 0
    if detector is not None:
        mode = "a" if done else "w"
        with (run_dir / "responses.jsonl").open(mode, encoding="utf-8", newline="\n") as handle:
            for index, (case, item) in enumerate(zip(cases, packs), start=1):
                stored = done.get(case.case_id)
                if stored is not None and not stored.get("error"):
                    response = ollama.Response(
                        text=stored["text"], thinking=stored.get("thinking", ""),
                        prompt_tokens=int(stored.get("prompt_tokens", 0)),
                        completion_tokens=int(stored.get("completion_tokens", 0)),
                        duration_ms=int(stored.get("duration_ms", 0)))
                    reports, _ = contract.parse(response.text)
                    responses.append(response)
                else:
                    response = detector.complete(item.system, item.user, schema)
                    responses.append(response)
                    failures += bool(response.error)
                    reports, bad = contract.parse(response.text)
                    rejects.extend({"case_id": case.case_id, "stage": reject.stage,
                                    "detail": reject.detail, "payload": reject.payload} for reject in bad)
                    handle.write(json.dumps({
                        "case_id": case.case_id, "text": response.text, "thinking": response.thinking,
                        "error": response.error, "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "duration_ms": response.duration_ms,
                    }, ensure_ascii=False) + "\n")
                    # Flushed per call, so the file is readable while a long run goes on.
                    handle.flush()
                    if not args.quiet:
                        print(f"[{index}/{len(packs)}] {case.case_id}: {len(reports)} finding(s), "
                              f"{response.duration_ms} ms" + (f"  ERROR {response.error}" if response.error else ""),
                              file=sys.stderr, flush=True)

                kept = contract.cap(contract.dedupe(reports), args.max_findings)
                dropped += len(reports) - len(kept)
                placements = scope.resolve(kept, case)
                out_of_scope.extend({"case_id": case.case_id, "file": p.report.file, "line": p.report.line,
                                     "type": p.report.type, "detail": p.detail, "title": p.report.title}
                                    for p in placements if not p.kept)
                kept = scope.apply(placements)
                predictions.extend(to_predictions(case.case_id, kept))
                decisions = anchor.resolve(kept, case, with_deletions=True)
                anchored.extend(to_predictions(case.case_id, anchor.apply(decisions)))
                anchors.extend({"case_id": case.case_id, "file": d.report.file, "claimed_line": d.report.line,
                                "line": d.line, "verdict": d.verdict, "detail": d.detail,
                                "quote": d.report.quote, "title": d.report.title} for d in decisions)

    runs.write_rows(run_dir / "predictions.jsonl", map(prediction_row, predictions))
    runs.write_rows(run_dir / "predictions.anchored.jsonl", map(prediction_row, anchored))
    runs.write_rows(run_dir / "out_of_scope.jsonl", out_of_scope)
    runs.write_rows(run_dir / "anchors.jsonl", anchors)
    runs.write_rows(run_dir / "rejects.jsonl", rejects)

    manifest = {
        "run_id": run_id, "stage": "detect", "repository": args.repo, "cases_file": str(cases_path),
        "cases": len(cases), "predictions": len(predictions), "anchored": len(anchored),
        "rejected": len(rejects), "call_failures": failures,
        "model": None if detector is None else detector.name, "model_build": build or None,
        "context_tokens": args.num_ctx, "prompt": prompt.digest(),
        "max_findings": args.max_findings, "dropped_over_cap": dropped,
        "dropped_out_of_scope": len(out_of_scope), "anchors": Counter(row["verdict"] for row in anchors),
        "reused_answers": len(done),
        # A run that lost its server part of the way through still writes every
        # file, because --resume reads them. It must not also look finished.
        "complete": detector is None or failures == 0,
        "sampling": {"temperature": args.temperature, "seed": args.seed, "think": args.think},
        "base_url": args.base_url if isinstance(detector, ollama.OllamaClient) else None,
        "budget": budget(packs, responses),
        **runs.provenance(argv),
    }
    runs.write_manifest(run_dir, manifest)

    if not args.quiet:
        cost = manifest["budget"]
        print(f"{args.repo}: {len(cases)} pull requests, {len(anchored)} claims")
        print(f"prompt tokens: estimated mean {cost['estimated_prompt_tokens_mean']} "
              f"max {cost['estimated_prompt_tokens_max']}"
              + (f" | measured mean {cost['measured_prompt_tokens_mean']} max {cost['measured_prompt_tokens_max']}"
                 if "measured_prompt_tokens_mean" in cost else ""))
        if done:
            print(f"reused {len(done)} answer(s) already on disk")
        if anchors:
            counts = Counter(row["verdict"] for row in anchors)
            print("anchors: " + "  ".join(f"{name}={n}" for name, n in counts.most_common()))
        if rejects:
            print(f"unreadable answers: {len(rejects)}  (see rejects.jsonl)")
        print(f"\n-> {run_dir}")
    if failures:
        print(f"\n!! {failures} of {len(packs)} calls failed; this run is INCOMPLETE. Run the same command "
              f"with --resume once the server is back; only the failed calls are asked again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
