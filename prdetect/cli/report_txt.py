"""Print a JSON report as plain text.

    python -m prdetect.cli.report_txt reports/genis_olcum_reposu/<run>.json

writes the `.txt` beside the JSON (or to `--out`). For every pull request: which
file and line each finding points at, what kind of finding it is, and -- when the
repository has an answer key -- which label it was assigned to, or that it was a
false alarm; then the labels no finding reached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

RUNG_NAMES = {"location": "konum", "family": "aile", "type": "tür"}
VERDICTS = {"true_positive": "doğru", "neutral": "puanlanmayan etiket", "false_alarm": "yanlış alarm",
            "unscored": "etiket yok"}


def _lines(start: int, end: int) -> str:
    return f"{start}" if end <= start else f"{start}-{end}"


def _short(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def render(report: dict) -> str:
    summary = report["summary"]
    out = [f"Repo      : {report['repository']}",
           f"Koşu      : {report['run']}",
           f"Model     : {report.get('model') or '-'}   prompt {report.get('prompt') or '-'}",
           f"PR        : {summary['pull_requests']} (bulgulu {summary['pull_requests_with_findings']})",
           f"Bulgu     : {summary['findings']}"]
    if report.get("labelled"):
        out.append(f"Doğruluk  : satır toleransı ±{report['tolerance']}")
        for name, card in summary.get("scores", {}).items():
            out.append(f"  {RUNG_NAMES.get(name, name):<6} TP={card['tp']:<3} FP={card['fp']:<3} FN={card['fn']:<3} "
                       f"P={card['precision']:.3f} R={card['recall']:.3f} F1={card['f1']:.3f}")
    else:
        out.append("Doğruluk  : bu repo için cevap anahtarı yok")
    out.append("")

    for pull in report["pull_requests"]:
        out.append("=" * 78)
        number = f"PR #{pull['pr_id']}" if pull.get("pr_id") is not None else pull["case_id"]
        out.append(f"{number}  {pull['title']}")
        if pull.get("url"):
            out.append(f"  {pull['url']}")
        if not pull["findings"]:
            out.append("  (bulgu yok)")
        for index, finding in enumerate(pull["findings"], start=1):
            out.append(f"  {index}. {finding['file']}:{_lines(finding['line'], finding['end_line'])}")
            out.append(f"     bulgu  : {finding['type']}  (güven {finding['confidence']:.2f})")
            out.append(f"     başlık : {_short(finding['title'])}")
            verdict = finding.get("verdict", "unscored")
            label = finding.get("assigned_to")
            if label:
                kind = "aynı" if label["same_type"] else f"farklı ({label['type']})"
                family = "aynı" if label["same_family"] else f"farklı ({label['family']})"
                out.append(f"     atandı : {label['finding_id']}  {label['file']}:{_lines(*label['lines'])}"
                           f"  [{VERDICTS[verdict]}; tür {kind}; aile {family}]")
                out.append(f"     etiket : {_short(label['title'])}")
            else:
                out.append(f"     atandı : {VERDICTS[verdict]}")
        for missed in pull.get("missed", []):
            out.append(f"  ! kaçırılan etiket: {missed['finding_id']}  {missed['type']}  "
                       f"{missed['file']}:{_lines(*missed['lines'])}")
            out.append(f"     etiket : {_short(missed['title'])}")
    out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a JSON report as plain text.")
    parser.add_argument("report", type=Path, help="a JSON report written by prdetect.cli.report")
    parser.add_argument("--out", type=Path, help="defaults to the report path with .txt")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    out = args.out or args.report.with_suffix(".txt")
    out.write_text(render(report), encoding="utf-8")
    print(f"{report['repository']}: {len(report['pull_requests'])} pull requests -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
