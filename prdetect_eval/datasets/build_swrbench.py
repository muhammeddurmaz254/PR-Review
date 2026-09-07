"""Export the SWRBench sample into the harness's case format.

These are real pull requests from eleven open-source Python projects, so unlike a
generated corpus there is no checkout to read: the record carries the pull
request's title and statement plus the diff of each of its commits, and nothing
else. That shape decides what can be built on it -- a detector here reviews the
hunks it is given, which is arm C of the architecture, and the one arm never
written.

Ground truth comes from ``swrbench_labels.json``, itself derived from the
records: the *resolving* diff's removed lines are the tightest statement of where
the defect was, and the *introducing* hunk is the region a reviewer would be
looking at. Both are kept -- region for localization, focus for the strict IoU.

``head_files`` is deliberately left empty. Filling it would mean cloning eleven
upstream repositories at their base commits, and every consumer already treats a
missing file as "not available" rather than "empty".

The code shown for a case is **the pull request's own commits**, for defective
and clean cases alike. Two earlier choices were wrong in opposite directions and
cancelled the measurement out: a defective case was shown the reviewer's own
snippet, which is the answer's neighbourhood, and a clean case was shown nothing
at all -- an empty diff and an empty file list. Zero false alarms over
twenty-five clean cases was therefore not a precision result, it was an empty
prompt. The resolving commit is in none of the twenty-five defective pull
requests, so their commits are the pull request as submitted: what a reviewer
reads before saying anything.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "datas" / "swrbench_analiz_50.jsonl"

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def commit_diff(record: dict) -> str:
    """The pull request's commits, in order, each marked with its sha.

    Line numbers inside a hunk are that commit's view of the file. Eleven pull
    requests touch one file from two commits, so the sha is printed rather than
    dropped: the numbering is per commit, and hiding that would make two
    different numberings look like one.
    """
    blocks = []
    for commit in record.get("pr_commits", []):
        text = (commit.get("diff_text") or "").strip()
        if not text:
            continue
        message = (commit.get("message") or "").split("\n")[0].strip()
        blocks.append(f"# commit {commit.get('sha', '')[:12]} {message}\n{text}")
    return "\n\n".join(blocks)


def touched(diff: str) -> dict[str, list[int]]:
    """Which lines each file gained, read from the diff rather than the labels.

    Taking these from the label ranges -- as this exporter first did -- hands the
    ``every_added_line`` baseline the answer and turns a trivial baseline into an
    oracle.
    """
    added: dict[str, set[int]] = {}
    filename = ""
    number = 0
    inside = False
    for line in diff.split("\n"):
        match = HUNK.match(line)
        if match:
            inside, number = True, int(match.group(1))
            continue
        if inside:
            mark = line[:1]
            if mark == "+":
                added.setdefault(filename, set()).add(number)
                number += 1
                continue
            if mark == "-":
                continue
            if mark == " " or line == "":
                number += 1
                continue
            inside = False
        if line.strip() and not line.startswith("# commit "):
            filename = line.strip()
    return {name: sorted(lines) for name, lines in sorted(added.items())}


def build(labels_path: Path, source_path: Path, out_path: Path) -> int:
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    if not source_path.exists():
        raise SystemExit(
            f"{source_path} is missing; it carries the pull requests' commit diffs "
            f"and is not in version control. Pass --source."
        )
    source = {}
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            source[record["instance_id"]] = record
    rows = []
    for case in document["cases"]:
        findings, names = [], []
        for index, finding in enumerate(case["findings"], start=1):
            name = finding["file"]
            names.append(name)
            findings.append({
                "finding_id": f"{case['case_id']}-f{index}",
                "type": finding["type"],
                "in_scope": bool(finding["in_scope"]),
                "required": bool(finding["in_scope"]),
                "role": "primary",
                "file": name,
                "start_line": finding["start_line"],
                "end_line": finding["end_line"],
                "focus_start": finding["focus_line"],
                "focus_end": finding["focus_end"],
                "anchor_rule": finding["span_kind"],
                "anchor_text": "",
                "title": finding.get("title", ""),
                "in_diff": True,
                "cross_file": False,
                "pure_deletion": False,
                "equivalent_locations": [],
                "severity": "", "cwe": "",
                "rationale": finding.get("title", ""),
            })

        record = source.get(case["case_id"])
        if record is None:
            raise SystemExit(f"{case['case_id']} has labels but no source record")
        diff = commit_diff(record)
        added_lines = touched(diff)

        rows.append({
            "case_id": case["case_id"],
            "pair_id": None,
            "variant": "buggy" if case["is_defective"] else "clean",
            "difficulty": "hard",
            "primary_type": case["primary_type"],
            "is_defective": bool(case["is_defective"]),
            "pr_title": case["title"],
            "pr_description": case.get("statement", ""),
            "changed_files": sorted(set(added_lines) | set(names)),
            "noise_files": [],
            "deleted_files": [],
            "added_lines": added_lines,
            "head_files": {},
            "diff": diff,
            "findings": findings,
            "distractors": [],
            "base_commit": case.get("base_commit", ""),
            "head_commit": "",
            "branch": case.get("repo", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export SWRBench into the harness case format.")
    parser.add_argument("--labels", type=Path, default=HERE / "swrbench_labels.json")
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="the raw SWRBench records, which carry the commit diffs")
    parser.add_argument("--out", type=Path, default=HERE / "swrbench.eval.jsonl")
    args = parser.parse_args(argv)
    print(f"Exported {build(args.labels, args.source, args.out)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
