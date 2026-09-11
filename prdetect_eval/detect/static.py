"""Layer 1: what standard analyzers newly say about a pull request.

The reviewer answers by reading, and some questions are not reading
questions: is a name this change deleted still used somewhere, does a call
still match the function's signature, can this value be None here. A model
asked them guesses -- nine of the best chain's eleven remaining false alarms
were claims whose truth hung on code the model was not shown. Analyzers
answer them by computation, over the whole repository, the same way in
every Python repository. Nothing here was fitted to a corpus.

Only what the pull request *introduces* is reported: the analyzers run on
the revision before and the revision after, and a diagnostic counts when
its kind became more frequent. Lines shift between revisions, so the match
is by (tool, rule, file, first line of the message), never by line number.
When a kind grew by k, the k instances on lines the change added are
preferred, because that is where a new instance most plausibly came from.

Rule selection was made before any result was read, and aims at defects,
not style: pyflakes (F), syntax (E9), bugbear (B), the bandit security rules
(S), blind except (BLE), naive datetimes (DTZ), pylint errors (PLE) and async
misuse (ASYNC). S101 -- "assert used" -- is left out: an assert in a test is
the normal case, the rule concerns code compiled with -O, and every pull
request that adds a test would trip it.

Type checking needs the repository's own dependencies to resolve imports;
without them every third-party call is Unknown and the checker goes quiet.
That is a requirement on whoever deploys this, not something a rule can fix.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping

RUFF_SELECT = "F,E9,B,S,BLE,DTZ,PLE,ASYNC"
RUFF_IGNORE = "S101"


@dataclass(frozen=True)
class Diagnostic:
    tool: str
    rule: str
    file: str
    line: int
    message: str

    @property
    def kind(self) -> tuple[str, str, str, str]:
        return (self.tool, self.rule, self.file, " ".join(self.message.split("\n")[0].split()))

    def as_dict(self) -> dict:
        return asdict(self)


def _relative(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def run_ruff(tree: str, ruff: str) -> list[Diagnostic]:
    out = subprocess.run([ruff, "check", "--isolated", "--select", RUFF_SELECT, "--ignore", RUFF_IGNORE,
                          "--output-format", "json", "--no-cache", "--exit-zero", tree],
                         capture_output=True, text=True, check=True).stdout
    return [Diagnostic("ruff", d["code"] or "?", _relative(d["filename"], tree), d["location"]["row"], d["message"])
            for d in json.loads(out or "[]")]


def run_pyright(tree: str, pyright: str, python: str) -> list[Diagnostic]:
    # From inside the tree: the checker takes its working directory as the
    # import root, and run from anywhere else it cannot resolve the
    # repository's own packages -- every project import is "unresolved" and
    # the checker sees nothing.
    out = subprocess.run([pyright, "--outputjson", "--pythonpath", python], cwd=tree,
                         capture_output=True, text=True).stdout
    document = json.loads(out or "{}")
    return [Diagnostic("pyright", d.get("rule") or d.get("severity", "?"), _relative(d["file"], tree),
                       d["range"]["start"]["line"] + 1, d["message"])
            for d in document.get("generalDiagnostics", []) if d.get("severity") in ("error", "warning")]


def introduced(base: Iterable[Diagnostic], head: Iterable[Diagnostic],
               added: Mapping[str, frozenset[int]] | None = None) -> list[Diagnostic]:
    """Head diagnostics whose kind the change made more frequent."""
    before = Counter(d.kind for d in base)
    after: dict[tuple, list[Diagnostic]] = defaultdict(list)
    for d in head:
        after[d.kind].append(d)
    added = added or {}
    out: list[Diagnostic] = []
    for kind, instances in after.items():
        grew = len(instances) - before.get(kind, 0)
        if grew <= 0:
            continue
        on_added = [d for d in instances if d.line in added.get(d.file, frozenset())]
        rest = [d for d in instances if d not in on_added]
        out += (on_added + rest)[:grew]
    return sorted(out, key=lambda d: (d.file, d.line, d.tool, d.rule))
