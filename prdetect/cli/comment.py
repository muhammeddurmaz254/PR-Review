"""Write a run's findings under its pull requests on Bitbucket, one table comment each.

    python -m prdetect.cli.comment --run <publish-run>           # preview only
    python -m prdetect.cli.comment --run <publish-run> --post    # create or update the comments

Without `--post` nothing leaves this machine: each comment is written to
`runs/<run>/comments/PR-<id>.md` to be read first. With `--post`, every pull
request gets one general comment -- its findings as a table of severity, kind,
file:line and title, or "Bulgu yok" when there are none -- and a pull request
that already has one from an earlier run has it updated in place.

"Bulgu yok" is written only by a complete run. When a model call or a
verification failed, a pull request without findings may just not have been
answered: it gets no comment, and an old one is left as it is.

A pull request whose source branch has moved since it was fetched is skipped:
its line numbers would point at other code. A pull request Bitbucket fails to
answer for is recorded as an error and the rest go on; a refused token stops the
command. Running it again is always safe: a comment already there is updated or
left as it is, never added twice. Every action is recorded, as it happens, in
`runs/<run>/comments.jsonl`, and the command exits 1 when any pull request was
skipped or failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from prdetect import paths
from prdetect.bitbucket import comment
from prdetect.bitbucket.client import DEFAULT_WORKSPACE, Bitbucket, BitbucketError
from prdetect.cli import runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a run's findings under its pull requests on Bitbucket.")
    parser.add_argument("--run", required=True, help="a run whose predictions.jsonl is published, normally publish")
    parser.add_argument("--post", action="store_true", help="write to Bitbucket; without it only previews are written")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--case", action="append", default=[], help="only these cases, e.g. PR-7 (repeatable)")
    parser.add_argument("--out", type=Path, default=paths.RUNS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run_dir = runs.run_dir(args.run, args.out)
    if not (run_dir / "predictions.jsonl").exists():
        raise SystemExit(f"{run_dir} has no predictions.jsonl")
    manifest = runs.read_manifest(run_dir)
    repo = manifest["repository"]
    cases = runs.cases_by_id(manifest)
    findings: dict[str, list[dict]] = defaultdict(list)
    for row in runs.read_rows(run_dir / "predictions.jsonl"):
        if row["case_id"] not in cases:
            raise SystemExit(f"{row['case_id']} is not a pull request of {repo}")
        findings[row["case_id"]].append(row)

    # A failed call leaves some pull request unanswered, and its silence is not "no findings".
    complete = (manifest.get("complete", True) and not manifest.get("call_failures")
                and not manifest.get("errors"))
    client = Bitbucket.from_env() if args.post else None
    previews = run_dir / "comments"
    previews.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    with (run_dir / "comments.jsonl").open("w", encoding="utf-8", newline="\n") as handle:

        def record(row: dict) -> None:
            log.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        for case_id, case in cases.items():
            if args.case and case_id not in args.case:
                continue
            pull_id = case.pull_request.get("id")
            if pull_id is None:
                record({"case_id": case_id, "action": "no pull request"})
                continue
            raw = comment.render(findings[case_id], case.head_commit, manifest.get("model"), run_dir.name,
                                 reviews_source=case.reviewable)
            (previews / f"{case_id}.md").write_text(raw + "\n", encoding="utf-8")
            row = {"case_id": case_id, "pull_request": pull_id, "findings": len(findings[case_id])}
            if not findings[case_id] and not complete:
                row["action"] = "incomplete run"
            elif client is None:
                row["action"] = "preview"
            else:
                try:
                    current = client.pull_request(args.workspace, repo, pull_id)["source"]["commit"]["hash"]
                    if not (current.startswith(case.head_commit) or case.head_commit.startswith(current)):
                        row |= {"action": "source moved", "fetched": case.head_commit, "current": current}
                    else:
                        action, comment_id = comment.upsert(client, args.workspace, repo, pull_id, raw)
                        row |= {"action": action, "comment_id": comment_id}
                except BitbucketError as error:
                    if error.status in (401, 403):
                        raise
                    row |= {"action": "error", "error": str(error)}
            record(row)
            if not args.quiet and row["action"] != "preview":
                detail = f"  {row['error']}" if row["action"] == "error" else ""
                print(f"PR #{pull_id}: {row['action']} ({row['findings']} findings){detail}",
                      file=sys.stderr, flush=True)

    counts = Counter(row["action"] for row in log)
    if not args.quiet:
        print(f"{repo}: " + ", ".join(f"{action} {n}" for action, n in counts.most_common())
              + (f" -> previews in {previews}" if client is None else ""))
        if counts["error"]:
            print("some pull requests failed; running the same command again is safe", file=sys.stderr)
        if counts["incomplete run"]:
            print(f"{counts['incomplete run']} pull requests without findings got no comment: the run is incomplete",
                  file=sys.stderr)
    return 1 if counts["source moved"] or counts["error"] or counts["incomplete run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
