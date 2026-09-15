"""Turn a Bitbucket pull request into a case.

One case per pull request, keyed `PR-<id>`. A case carries the pull request's
title and description, its diff, the new-side lines it adds, the files it changed
as they are at its source commit, and every other file at that commit as context:
the verifier's tools and the facts read the repository, not only the diff.

Labels come from the repository's answer key, keyed by pull request id. A pull
request the key does not name is fetched unlabelled: the report still lists every
finding, and there is nothing to score it against. A key written for another
commit is refused, because its line numbers would point at code that has moved.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

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
    """The pull requests an answer key labels, by id; empty when there is no key."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["pull_requests"]


def build_case(pull: dict, diff: str, tree: dict[str, str], key: dict | None) -> dict:
    """One line of a cases file, from a pull request, its diff, its tree and its labels."""
    source = pull["source"]["commit"]["hash"]
    if key and not key["head_commit"].startswith(source):
        raise SystemExit(f"PR #{pull['id']}: the answer key was written for {key['head_commit'][:12]}, "
                         f"the pull request is now at {source}; its line numbers no longer apply")
    case_id = f"PR-{pull['id']}"
    changed, deleted, added = parse_diff(diff)
    missing = [name for name in changed if name not in tree]
    if missing:
        raise SystemExit(f"{case_id}: {missing} changed but not in the tree at {source}")
    findings = [{"finding_id": f"{case_id}-f{index}", **finding}
                for index, finding in enumerate((key or {}).get("findings", []), start=1)]
    return {
        "case_id": case_id,
        "labelled": key is not None,
        "is_defective": bool((key or {}).get("is_defective")),
        "pr_title": pull.get("title") or "",
        "pr_description": pull.get("description") or "",
        "changed_files": changed,
        "deleted_files": deleted,
        "added_lines": added,
        "head_files": {name: tree[name] for name in changed},
        "context_files": {name: text for name, text in sorted(tree.items()) if name not in changed},
        "diff": diff,
        "findings": findings,
        "base_commit": pull["destination"]["commit"]["hash"],
        "head_commit": source,
        "pull_request": {"id": pull["id"], "url": pull["links"]["html"]["href"],
                         "source_branch": pull["source"]["branch"]["name"],
                         "destination_branch": pull["destination"]["branch"]["name"]},
    }
