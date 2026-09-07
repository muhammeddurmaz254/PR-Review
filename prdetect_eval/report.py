"""Markdown rendering of an evaluation run.

The layout follows what the numbers are for: the cascade shows how much of the
score is localization loss, the baseline table shows whether the score means
anything at all, and the stage table shows which layer to spend a week on.
"""
from __future__ import annotations

from typing import Sequence


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    widths = [len(str(header)) for header in headers]
    text_rows = [[str(cell) for cell in row] for row in rows]
    for row in text_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = [
        "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in text_rows]
    return lines + [""]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(run_id: str, config: dict, result: dict, baselines: dict[str, dict] | None = None,
           stages: list[dict] | None = None, notes: Sequence[str] = ()) -> str:
    primary = result["primary"]
    lines = [
        f"# Evaluation run `{run_id}`",
        "",
        f"Source **{config.get('source', 'unknown')}** | cases **{config.get('cases', '?')}** | "
        f"tolerance **k={primary['tolerance']}** | type mode **{primary['type_mode']}** | "
        f"threshold **{primary['threshold']}**",
        "",
        "## Primary metric (required in-scope labels, k=3, exact type)",
        "",
    ]
    lines += _table(
        ["TP", "FP", "FN", "precision", "recall", "F1", "neutral (optional / out-of-scope)"],
        [[primary["tp"], primary["fp"], primary["fn"], _pct(primary["precision"]),
          _pct(primary["recall"]), _pct(primary["f1"]),
          f"{primary['neutral_optional']} / {primary['neutral_out_of_scope']}"]],
    )

    lines += ["## Localization cascade", "",
              "The drop from the first row to the last is localization loss, not detection loss.", ""]
    lines += _table(
        ["rung", "TP", "FP", "FN", "precision", "recall", "F1"],
        [[name, card["tp"], card["fp"], card["fn"], _pct(card["precision"]), _pct(card["recall"]), _pct(card["f1"])]
         for name, card in result["cascade"].items()],
    )

    lines += ["## Pull-request level", "",
              "Whether the pull request gets a comment at all. Balanced accuracy is the "
              "figure to read: it is 50% for any strategy that ignores the input, on any "
              "class balance, where plain accuracy rewards a corpus of mostly-clean cases.", ""]
    lines += _table(
        ["universe", "TP", "FP", "TN", "FN", "precision", "recall", "specificity",
         "F1", "accuracy", "balanced acc."],
        [[name, card["tp"], card["fp"], card["tn"], card["fn"],
          _pct(card["precision"]), _pct(card["recall"]), _pct(card["specificity"]),
          _pct(card["f1"]), _pct(card["accuracy"]), _pct(card["balanced_accuracy"])]
         for name, card in (("in-scope", result["pr_level_in_scope"]), ("all types", result["pr_level_all"]))],
    )

    lines += ["## Pairwise accuracy", "",
              "A feature counts only when the defective variant is caught **and** its clean twin stays silent.", ""]
    lines += _table(
        ["universe", "pairs", "correct", "accuracy", "missed", "noisy clean twin"],
        [[name, card["pairs"], card["correct"], _pct(card["accuracy"]),
          len(card["missed"]), len(card["noisy_clean_twin"])]
         for name, card in (("in-scope", result["pairwise_in_scope"]), ("all types", result["pairwise_all"]))],
    )

    alarms = result["false_alarms"]
    lines += ["## False alarms", ""]
    lines += _table(
        ["case kind", "cases", "unmatched reports", "per PR"],
        [[name, alarms[name]["cases"], alarms[name]["total"], f"{alarms[name]['per_pr']:.2f}"]
         for name in ("all", "defective", "clean_twin", "standalone_trap")]
        + [["-> of which on a distractor", "", alarms["on_distractor"], ""]],
    )

    lines += ["## Breakdown", ""]
    for key, title in (("by_type", "By defect type"), ("by_family", "By family"), ("by_difficulty", "By difficulty")):
        lines += [f"### {title}", ""]
        lines += _table(
            ["group", "TP", "FN", "recall"],
            [[name, row["tp"], row["fn"], _pct(row["recall"])] for name, row in result[key].items()],
        )

    lines += ["## Threshold sweep", ""]
    lines += _table(
        ["threshold", "TP", "FP", "FN", "precision", "recall", "F1", "FP/PR", "pairwise"],
        [[row["threshold"], row["tp"], row["fp"], row["fn"], _pct(row["precision"]), _pct(row["recall"]),
          _pct(row["f1"]), f"{row['fp_per_pr']:.2f}", _pct(row["pairwise"])]
         for row in result["threshold_sweep"]],
    )

    if stages:
        lines += ["## Stage loss table", ""]
        lines += _table(
            ["stage", "entered", "left", "TP kept", "TP lost", "FP", "FP added"],
            [[row["stage"], row["entered"], row["left"], row["tp_kept"], row["tp_lost"], row["fp"], row["fp_added"]]
             for row in stages],
        )

    if baselines:
        lines += ["## Trivial baselines", "",
                  "Any number above is meaningless unless it clears these. "
                  "`hot_spot_memoriser` and `every_added_line` read the answer key or report "
                  "everything; they are ceilings, not competitors.", ""]
        rows = []
        for name, card in baselines.items():
            rows.append([name, card["predictions"], card["tp"], card["fp"], card["fn"],
                         _pct(card["precision"]), _pct(card["recall"]), _pct(card["f1"]),
                         _pct(card["pairwise"]), f"{card['fp_per_pr']:.2f}",
                         _pct(card["pr_level"]["balanced_accuracy"])])
        lines += _table(
            ["baseline", "reports", "TP", "FP", "FN", "precision", "recall", "F1", "pairwise",
             "FP/PR", "PR-level bal. acc."], rows
        )

    if notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in notes] + [""]
    return "\n".join(lines) + "\n"


def console(result: dict, baselines: dict[str, dict] | None = None) -> str:
    """Compact one-screen summary for the terminal."""
    primary = result["primary"]
    out = [
        f"k={primary['tolerance']} {primary['type_mode']}  "
        f"TP={primary['tp']} FP={primary['fp']} FN={primary['fn']}  "
        f"P={_pct(primary['precision'])} R={_pct(primary['recall'])} F1={_pct(primary['f1'])}",
        f"pairwise(in-scope)={_pct(result['pairwise_in_scope']['accuracy'])}  "
        f"FP/PR={result['false_alarms']['all']['per_pr']:.2f}",
        "by type: " + "  ".join(f"{name}={_pct(row['recall'])}" for name, row in result["by_type"].items()),
    ]
    flagging = result["pr_level_in_scope"]
    out.insert(1, (
        f"PR-level  caught={flagging['tp']}/{flagging['tp'] + flagging['fn']} "
        f"quiet={flagging['tn']}/{flagging['tn'] + flagging['fp']}  "
        f"balanced acc={_pct(flagging['balanced_accuracy'])} "
        f"(0.5 = ignoring the input)"
    ))
    if baselines:
        best = max(baselines.items(), key=lambda item: item[1]["recall"])
        out.append(f"best trivial baseline: {best[0]} R={_pct(best[1]['recall'])} F1={_pct(best[1]['f1'])}")
        honest = {k: v for k, v in baselines.items() if not v["leaky"]}
        if honest:
            top = max(honest.items(), key=lambda item: item[1]["pr_level"]["balanced_accuracy"])
            out.append(f"best trivial at PR level: {top[0]} "
                       f"balanced acc={_pct(top[1]['pr_level']['balanced_accuracy'])}")
    return "\n".join(out)
