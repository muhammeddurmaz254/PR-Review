"""Fetch a repository's pull requests from Bitbucket as cases.

    python -m prdetect.cli.fetch --repo genis_olcum_reposu

writes `data/cases/<repo>.jsonl`, the file every later stage reads. Labels come
from `data/answer_keys/<repo>.json` when it exists. Nothing is written to
Bitbucket.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.bitbucket.client import DEFAULT_WORKSPACE, Bitbucket
from prdetect.bitbucket.fetch import build_case, load_answer_key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a repository's Bitbucket pull requests as cases.")
    parser.add_argument("--repo", required=True, help="repository slug, e.g. genis_olcum_reposu")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--state", default="OPEN", choices=("OPEN", "MERGED", "DECLINED", "SUPERSEDED"))
    parser.add_argument("--answer-key", type=Path, help="defaults to data/answer_keys/<repo>.json")
    parser.add_argument("--out", type=Path, help="defaults to data/cases/<repo>.jsonl")
    args = parser.parse_args(argv)

    client = Bitbucket.from_env()
    answer_key = load_answer_key(args.answer_key or paths.answer_key_file(args.repo))
    pulls = client.pull_requests(args.workspace, args.repo, args.state)
    if not pulls:
        raise SystemExit(f"{args.workspace}/{args.repo}: no {args.state} pull requests")
    unknown = sorted(set(answer_key) - {str(pull["id"]) for pull in pulls}, key=int)
    if unknown:
        print(f"note: the answer key names PRs that are not {args.state}: {', '.join(unknown)}", file=sys.stderr)

    rows = []
    for pull in pulls:
        diff = client.diff(args.workspace, args.repo, pull)
        tree = client.tree(args.workspace, args.repo, pull["source"]["commit"]["hash"])
        rows.append(build_case(pull, diff, tree, answer_key.get(str(pull["id"]))))
        print(f"  PR #{pull['id']}: {len(rows[-1]['changed_files'])} changed, "
              f"{len(rows[-1]['context_files'])} context, {len(rows[-1]['findings'])} labels", file=sys.stderr)

    out = args.out or paths.cases_file(args.repo)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    labelled = sum(row["labelled"] for row in rows)
    print(f"{args.workspace}/{args.repo}: {len(rows)} pull requests ({labelled} labelled) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
