"""Oracle-judgment ablation: the ceiling of candidate enumeration.

No model runs. The only question is whether the enumerators put the true defect
in front of a model at all -- if a label is never enumerated, no detector,
prompt or quantization can ever recover it. That upper bound is the deliverable
of phase 0a, and it costs no GPU time to obtain.

Two numbers matter and they are reported side by side:

* **coverage** -- the share of required in-scope labels lying inside some
  candidate region of the right type. This is the true ceiling, because the
  model localizes freely inside the region it is shown.
* **cost** -- candidates and context lines per pull request. Coverage bought by
  widening regions is not progress, and this column makes that visible.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

import candidates as candidate_module
import report as report_module
from adapters import DEFAULT_EVAL, load_cases, write_predictions
from schema import IN_SCOPE_TYPES, Candidate, Case, Label

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TOLERANCES = (0, 3, 5, 10)


def _covers(candidate: Candidate, label: Label, same_type: bool) -> bool:
    if same_type and candidate.type != label.type:
        return False
    return any(candidate.region.distance(span) == 0 for span in label.spans)


def _focus_distance(candidate: Candidate, label: Label) -> int | None:
    distances = [
        distance for span in label.spans
        if (distance := candidate.focus.distance(span)) is not None
    ]
    return min(distances) if distances else None


def ceiling(cases: Sequence[Case], by_case: dict[str, list[Candidate]]) -> dict:
    """Per-type coverage of the required in-scope labels."""
    buckets: dict[str, dict] = defaultdict(lambda: {
        "labels": 0, "file": 0, "file_type": 0, "region": 0, "region_any_type": 0,
        **{f"focus_{k}": 0 for k in TOLERANCES}, "missed": [],
    })
    for case in cases:
        found = by_case.get(case.case_id, [])
        for label in case.scored_labels:
            row = buckets[label.type]
            row["labels"] += 1
            same_file = [c for c in found if c.focus.file == label.span.file or c.region.file == label.span.file]
            typed = [c for c in same_file if c.type == label.type]
            row["file"] += bool(same_file)
            row["file_type"] += bool(typed)
            row["region_any_type"] += any(_covers(c, label, same_type=False) for c in found)
            covered = any(_covers(c, label, same_type=True) for c in found)
            row["region"] += covered
            for tolerance in TOLERANCES:
                row[f"focus_{tolerance}"] += any(
                    (distance := _focus_distance(c, label)) is not None and distance <= tolerance
                    for c in typed
                )
            if not covered:
                row["missed"].append({
                    "finding_id": label.finding_id, "case_id": case.case_id,
                    "file": label.span.file, "line": label.span.start_line, "anchor": label.anchor_text,
                })
    return dict(sorted(buckets.items()))


def cost(cases: Sequence[Case], by_case: dict[str, list[Candidate]]) -> dict:
    """What the enumerators would hand the model, per pull request."""
    def summarise(subset: Sequence[Case]) -> dict:
        counts = [len(by_case.get(case.case_id, [])) for case in subset]
        context = []
        for case in subset:
            lines: set[tuple[str, int]] = set()
            for candidate in by_case.get(case.case_id, []):
                lines.update((candidate.region.file, line) for line in candidate.region.lines)
            context.append(len(lines))
        if not counts:
            return {"cases": 0}
        return {
            "cases": len(subset), "total": sum(counts),
            "per_pr_mean": round(sum(counts) / len(counts), 2),
            "per_pr_median": median(counts), "per_pr_max": max(counts),
            "context_lines_mean": round(sum(context) / len(context), 1),
            "context_lines_max": max(context),
        }

    return {
        "all": summarise(cases),
        "defective": summarise([case for case in cases if case.is_defective]),
        "clean_twin": summarise([case for case in cases if not case.is_defective and case.pair_id]),
        "standalone_trap": summarise([case for case in cases if not case.is_defective and not case.pair_id]),
        "by_type": {
            defect_type: {
                "total": sum(1 for group in by_case.values() for c in group if c.type == defect_type),
                "per_pr": round(
                    sum(1 for group in by_case.values() for c in group if c.type == defect_type) / len(cases), 2
                ),
            }
            for defect_type in sorted(IN_SCOPE_TYPES)
        },
        "by_detector": {
            detector: sum(1 for group in by_case.values() for c in group if c.detector == detector)
            for detector in sorted({c.detector for group in by_case.values() for c in group})
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(run_id: str, coverage: dict, costs: dict, cases: Sequence[Case]) -> str:
    total_labels = sum(row["labels"] for row in coverage.values())
    total_region = sum(row["region"] for row in coverage.values())
    lines = [
        f"# Candidate enumeration ceiling `{run_id}`",
        "",
        "Oracle judgment: no model runs. A label the enumerators never surface is "
        "unreachable for every downstream stage, so this is the hard upper bound "
        "on recall for phases 0b and later.",
        "",
        f"## Headline: **{_pct(total_region / total_labels if total_labels else 0)}** "
        f"({total_region}/{total_labels} required in-scope labels enumerated)",
        "",
        "## Coverage by type",
        "",
        "`region` is the ceiling -- the label lies inside a candidate of the right type. "
        "`focus k` is how close the enumerator points on its own, before the model narrows it.",
        "",
    ]
    lines += report_module._table(
        ["type", "arm", "labels", "file", "file+type", "region", "focus k=0", "k=3", "k=5", "k=10"],
        [[
            defect_type, candidate_module.ARM.get(defect_type, ""), row["labels"],
            _pct(row["file"] / row["labels"]), _pct(row["file_type"] / row["labels"]),
            _pct(row["region"] / row["labels"]),
            *[_pct(row[f"focus_{k}"] / row["labels"]) for k in TOLERANCES],
        ] for defect_type, row in coverage.items()],
    )

    lines += ["## Cost per pull request", "",
              "Coverage bought by widening regions shows up here, not above.", ""]
    lines += report_module._table(
        ["case kind", "cases", "candidates", "per PR (mean)", "median", "max", "context lines (mean)", "max"],
        [[name, costs[name]["cases"], costs[name]["total"], costs[name]["per_pr_mean"],
          costs[name]["per_pr_median"], costs[name]["per_pr_max"],
          costs[name]["context_lines_mean"], costs[name]["context_lines_max"]]
         for name in ("all", "defective", "clean_twin", "standalone_trap") if costs[name].get("cases")],
    )

    lines += ["### By type", ""]
    lines += report_module._table(
        ["type", "candidates", "per PR"],
        [[name, row["total"], row["per_pr"]] for name, row in costs["by_type"].items()],
    )
    lines += ["### By detector", ""]
    lines += report_module._table(
        ["detector", "candidates"], [[name, count] for name, count in costs["by_detector"].items()],
    )

    missed = [row for group in coverage.values() for row in group["missed"]]
    lines += ["## Labels no enumerator reached", ""]
    if missed:
        lines += report_module._table(
            ["finding", "case", "file", "line", "anchor"],
            [[row["finding_id"], row["case_id"], row["file"], row["line"], f"`{row['anchor'].strip()[:60]}`"]
             for row in missed],
        )
        lines += ["These are the ceiling. Revise the enumerators before spending GPU time.", ""]
    else:
        lines += ["None. Every required in-scope label is enumerated, so the ceiling is not "
                  "the constraint at this corpus size; the next bound is detector judgment.", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the ceiling of candidate enumeration.")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--types", nargs="*", default=None, help="restrict to these defect types")
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.eval)
    by_case = {case.case_id: candidate_module.enumerate_case(case, args.types) for case in cases}
    coverage, costs = ceiling(cases, by_case), cost(cases, by_case)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{stamp}_ceiling"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    flat = [candidate for group in by_case.values() for candidate in group]
    write_predictions(run_dir / "candidates_as_predictions.jsonl", [c.as_prediction() for c in flat])
    (run_dir / "candidates.jsonl").write_text("".join(
        json.dumps({
            "case_id": c.case_id, "type": c.type, "detector": c.detector, "symbol": c.symbol,
            "focus": {"file": c.focus.file, "start_line": c.focus.start_line, "end_line": c.focus.end_line},
            "region": {"file": c.region.file, "start_line": c.region.start_line, "end_line": c.region.end_line},
            "evidence": c.evidence,
        }, ensure_ascii=False, separators=(",", ":")) + "\n" for c in flat
    ), encoding="utf-8", newline="\n")
    (run_dir / "ceiling.json").write_text(
        json.dumps({
            "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": str(args.eval), "cases": len(cases), "types": args.types or sorted(IN_SCOPE_TYPES),
            "coverage": coverage, "cost": costs,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    markdown = render(run_id, coverage, costs, cases)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")

    if not args.quiet:
        total_labels = sum(row["labels"] for row in coverage.values())
        total_region = sum(row["region"] for row in coverage.values())
        print(f"ceiling (region, exact type): {total_region}/{total_labels} = "
              f"{_pct(total_region / total_labels if total_labels else 0)}")
        for defect_type, row in coverage.items():
            print(f"  {defect_type:16} region={_pct(row['region'] / row['labels'])} "
                  f"focus k=3 {_pct(row['focus_3'] / row['labels'])}  ({row['labels']} labels)")
        print(f"candidates/PR: mean {costs['all']['per_pr_mean']} max {costs['all']['per_pr_max']}  "
              f"context lines/PR: mean {costs['all']['context_lines_mean']}")
        print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
