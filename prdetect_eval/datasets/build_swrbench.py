"""Export the SWRBench sample into the harness's case format.

These are real pull requests from eleven open-source Python projects, so unlike a
generated corpus there is no checkout to read: the record carries the pull
request's title and statement plus diff snippets, and nothing else. That shape
decides what can be built on it -- a detector here reviews the hunk it is given,
which is arm C of the architecture, and the one arm never written.

Ground truth comes from ``swrbench_labels.json``, itself derived from the
records: the *resolving* diff's removed lines are the tightest statement of where
the defect was, and the *introducing* hunk is the region a reviewer would be
looking at. Both are kept -- region for localization, focus for the strict IoU.

``head_files`` is deliberately left empty. Filling it would mean cloning eleven
upstream repositories at their base commits, and every consumer already treats a
missing file as "not available" rather than "empty".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent


def build(labels_path: Path, out_path: Path) -> int:
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = []
    for case in document["cases"]:
        findings, touched, names = [], {}, []
        for index, finding in enumerate(case["findings"], start=1):
            name = finding["file"]
            names.append(name)
            touched.setdefault(name, []).extend(
                range(finding["start_line"], finding["end_line"] + 1)
            )
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

        rows.append({
            "case_id": case["case_id"],
            "pair_id": None,
            "variant": "buggy" if case["is_defective"] else "clean",
            "difficulty": "hard",
            "primary_type": case["primary_type"],
            "is_defective": bool(case["is_defective"]),
            "pr_title": case["title"],
            "pr_description": case.get("statement", ""),
            "changed_files": sorted(set(names)),
            "noise_files": [],
            "deleted_files": [],
            "added_lines": {name: sorted(set(lines)) for name, lines in touched.items()},
            "head_files": {},
            "diff": "\n\n".join(f.get("review_snippet", "") for f in case["findings"]),
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
    parser.add_argument("--out", type=Path, default=HERE / "swrbench.eval.jsonl")
    args = parser.parse_args(argv)
    print(f"Exported {build(args.labels, args.out)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
