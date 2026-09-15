"""Build a repository's cases from its open Bitbucket pull requests.

    python datasets/fetch_bitbucket.py --repo genis_olcum_reposu

writes `datasets/genis_olcum_reposu.eval.jsonl`, the file every stage reads, so
`run_detect.py --dataset genis_olcum_reposu` and everything after it run on what
Bitbucket holds. Nothing is written to Bitbucket.

One case per pull request, keyed `PR-<id>`. A case carries the pull request's
title and description, its diff, the new-side lines it adds, the changed files
at the source commit, and every other file at that commit as context -- the
verifier's tools and the facts read the repository, not only the diff.

Labels come from an answer key, `datasets/answer_keys/<repo>/labels_by_pr.json`,
keyed by pull request id. It is never shown to the model. A repository without
one is fetched unlabelled: the report still lists every finding, and scoring has
nothing to score. A key written for another commit is refused, because its line
numbers would point at code that is no longer there.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitbucket import DEFAULT_WORKSPACE, Bitbucket

HERE = Path(__file__).resolve().parent
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_diff(diff: str) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Changed paths (new side), deleted paths, and the new-side lines each file adds."""
    changed: list[str] = []
    deleted: list[str] = []
    added: dict[str, set[int]] = {}
    filename, number, inside = "", 0, False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            filename, inside = line.partition(" b/")[2].strip(), False
            changed.append(filename)
            continue
        if line.startswith("deleted file mode") and filename:
            deleted.append(changed.pop())
            continue
        if line.startswith("rename to "):
            changed[-1] = filename = line[len("rename to "):].strip()
            continue
        match = HUNK.match(line)
        if match:
            inside, number = True, int(match.group(1))
            continue
        if not inside:
            continue
        mark = line[:1]
        if mark == "+":
            added.setdefault(filename, set()).add(number)
            number += 1
        elif mark == "-":
            continue
        elif mark == " " or line == "":
            number += 1
        else:
            inside = False
    return sorted(set(changed)), sorted(set(deleted)), {name: sorted(lines) for name, lines in sorted(added.items())}


def load_answer_key(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["pull_requests"]


def build_case(pull: dict, diff: str, tree: dict[str, str], key: dict | None) -> dict:
    source = pull["source"]["commit"]["hash"]
    if key and not key["head_commit"].startswith(source):
        raise SystemExit(f"PR #{pull['id']}: the answer key was written for {key['head_commit'][:12]}, "
                         f"the pull request is now at {source}; its line numbers no longer apply")
    case_id = f"PR-{pull['id']}"
    changed, deleted, added = parse_diff(diff)
    missing = [name for name in changed if name not in tree]
    if missing:
        raise SystemExit(f"{case_id}: {missing} changed but not in the tree at {source}")
    findings = []
    for index, finding in enumerate((key or {}).get("findings", []), start=1):
        finding = {k: v for k, v in finding.items() if k != "source_finding_id"}
        findings.append({"finding_id": f"{case_id}-f{index}", **finding})
    return {
        "case_id": case_id,
        "pair_id": (key or {}).get("pair_id"),
        "variant": (key or {}).get("variant", "unlabelled"),
        "difficulty": (key or {}).get("difficulty", ""),
        "primary_type": (key or {}).get("primary_type"),
        "is_defective": bool((key or {}).get("is_defective")),
        "labelled": key is not None,
        "pr_title": pull.get("title") or "",
        "pr_description": pull.get("description") or "",
        "changed_files": changed,
        "noise_files": [],
        "deleted_files": deleted,
        "added_lines": added,
        "head_files": {name: tree[name] for name in changed},
        "context_files": {name: text for name, text in sorted(tree.items()) if name not in changed},
        "diff": diff,
        "findings": findings,
        "distractors": (key or {}).get("distractors", []),
        "base_commit": pull["destination"]["commit"]["hash"],
        "head_commit": source,
        "branch": pull["source"]["branch"]["name"],
        "pull_request": {"id": pull["id"], "url": pull["links"]["html"]["href"],
                         "source_branch": pull["source"]["branch"]["name"],
                         "destination_branch": pull["destination"]["branch"]["name"]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a repository's cases from its Bitbucket pull requests.")
    parser.add_argument("--repo", required=True, help="repository slug, e.g. genis_olcum_reposu")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--state", default="OPEN", choices=("OPEN", "MERGED", "DECLINED", "SUPERSEDED"))
    parser.add_argument("--answer-key", type=Path, help="defaults to datasets/answer_keys/<repo>/labels_by_pr.json")
    parser.add_argument("--out", type=Path, help="defaults to datasets/<repo>.eval.jsonl")
    args = parser.parse_args(argv)

    client = Bitbucket.from_env()
    answer_key = load_answer_key(args.answer_key or HERE / "answer_keys" / args.repo / "labels_by_pr.json")
    pulls = client.pull_requests(args.workspace, args.repo, args.state)
    if not pulls:
        raise SystemExit(f"{args.workspace}/{args.repo}: no {args.state} pull requests")
    unknown = sorted(set(answer_key) - {str(pull["id"]) for pull in pulls}, key=int)
    if answer_key and unknown:
        print(f"note: the answer key names PRs that are not {args.state}: {', '.join(unknown)}", file=sys.stderr)

    rows = []
    for pull in pulls:
        diff = client.diff(args.workspace, args.repo, pull)
        tree = client.tree(args.workspace, args.repo, pull["source"]["commit"]["hash"])
        rows.append(build_case(pull, diff, tree, answer_key.get(str(pull["id"]))))
        print(f"  PR #{pull['id']}: {len(rows[-1]['changed_files'])} changed, "
              f"{len(rows[-1]['context_files'])} context, {len(rows[-1]['findings'])} labels", file=sys.stderr)

    out = args.out or HERE / f"{args.repo}.eval.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    labelled = sum(row["labelled"] for row in rows)
    print(f"{args.workspace}/{args.repo}: {len(rows)} pull requests ({labelled} labelled) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
