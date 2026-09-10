"""Export the demo_repo corpus into the harness's case format.

The corpus is a real git repository: one branch per pull request, `main` as the
base. Ground truth lives beside it in ``demo_repo_labels.json`` because the
repository itself only records file and class -- the line spans were added
afterwards and are reviewed by hand.

Two details are corpus-specific and would silently corrupt every metric if they
were missed.

``beklentiler.json`` is the answer key. It sits on ``main`` and every branch
deletes it, so it appears in every diff as a 367-line removal. It is excluded
from changed files, from the diff and from the context the detector is shown.

The spans in the label file are *regions* -- the whole function containing the
defect -- because a detector that points anywhere inside the relevant code has
localized it. ``focus`` keeps the exact line for the tighter IoU column.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect import pack
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout


def changed_files(repo: Path, base: str, branch: str, exclude: Sequence[str]) -> list[str]:
    names = git(repo, "diff", "--name-only", base, branch).split("\n")
    return [name for name in names if name and name not in exclude]


def added_lines(repo: Path, base: str, branch: str, filename: str) -> list[int]:
    """One-based head line numbers the branch adds or rewrites."""
    diff = git(repo, "diff", "--no-ext-diff", "--unified=0", base, branch, "--", filename)
    lines: list[int] = []
    for row in diff.split("\n"):
        match = HUNK.match(row)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            lines.extend(range(start, start + count))
    return lines


def head_file(repo: Path, branch: str, filename: str) -> str | None:
    result = subprocess.run(
        ("git", "show", f"{branch}:{filename}"), cwd=repo, text=True, capture_output=True
    )
    return result.stdout if result.returncode == 0 else None


def build(labels_path: Path, out_path: Path) -> int:
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    repo = (ROOT / document["repo_path"]).resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a git repository")
    base = document["base_branch"]
    exclude = list(document.get("exclude_paths", []))

    rows = []
    for case in document["cases"]:
        branch = case["branch"]
        names = changed_files(repo, base, branch, exclude)
        touched, contents = {}, {}
        for name in names:
            touched[name] = added_lines(repo, base, branch, name)
            body = head_file(repo, branch, name)
            if body is not None:
                contents[name] = body
        diff = git(repo, "diff", base, branch, "--", *names) if names else ""

        findings = []
        for index, finding in enumerate(case["findings"], start=1):
            findings.append({
                "finding_id": f"{case['case_id']}-f{index}",
                "type": finding["type"],
                # Only code is put in front of the model, so a label in a
                # document is unreachable by construction and is not scored.
                "in_scope": bool(finding["in_scope"]) and pack.is_code(finding["file"]),
                "required": bool(finding["in_scope"]) and pack.is_code(finding["file"]),
                "role": "primary",
                "file": finding["file"],
                "start_line": finding["start_line"],
                "end_line": finding["end_line"],
                "focus_start": finding["focus_line"],
                "focus_end": finding.get("focus_end", finding["focus_line"]),
                "anchor_rule": finding["span_kind"],
                "anchor_text": finding.get("anchor_text", ""),
                "title": finding.get("title", ""),
                "in_diff": True,
                "cross_file": bool(finding.get("cross_file")),
                "pure_deletion": bool(finding.get("pure_deletion")),
                "equivalent_locations": [],
                "severity": "", "cwe": "",
                "rationale": finding.get("rationale", ""),
            })

        rows.append({
            "case_id": case["case_id"],
            # A trap is the harmless twin of a real case on the same surface, so
            # pairwise accuracy -- caught the defect and stayed quiet on the twin --
            # is measurable for the six that have one.
            "pair_id": case.get("pair_id"),
            "variant": "buggy" if case["is_defective"] else "clean",
            "difficulty": "hard" if any(f.get("cross_file") for f in case["findings"]) else "medium",
            "primary_type": case["primary_type"],
            "is_defective": bool(case["is_defective"]),
            "pr_title": case["title"],
            "pr_description": "",
            "changed_files": names,
            "noise_files": [],
            "deleted_files": [],
            "added_lines": touched,
            "head_files": contents,
            "diff": diff,
            "findings": findings,
            "distractors": [],
            "base_commit": git(repo, "rev-parse", base).strip(),
            "head_commit": git(repo, "rev-parse", branch).strip(),
            "branch": branch,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export demo_repo into the harness case format.")
    parser.add_argument("--labels", type=Path, default=HERE / "demo_repo_labels.json")
    parser.add_argument("--out", type=Path, default=HERE / "demo_repo.eval.jsonl")
    args = parser.parse_args(argv)
    count = build(args.labels, args.out)
    print(f"Exported {count} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
