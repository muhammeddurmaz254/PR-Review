"""The two repositories the tests read, as fetched from Bitbucket.

    python datasets/fetch_bitbucket.py --repo dar_olcum_reposu
    python datasets/fetch_bitbucket.py --repo genis_olcum_reposu

A test that needs one skips with that command when it has not been fetched.
Cases are keyed by pull request (`PR-7`); a test that means one particular
defect names the case it was published from, and the answer key translates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters import load_cases  # noqa: E402
from schema import Case  # noqa: E402

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
NARROW = "dar_olcum_reposu"
WIDE = "genis_olcum_reposu"
REPOSITORIES = (NARROW, WIDE)


def corpus(repo: str) -> Path:
    path = DATASETS / f"{repo}.eval.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} is not fetched: python datasets/fetch_bitbucket.py --repo {repo}")
    return path


def cases(repo: str) -> list[Case]:
    return load_cases(corpus(repo))


def case_id(repo: str, source_case: str) -> str:
    """The pull request a case was published as, from the answer key."""
    key = DATASETS / "answer_keys" / repo / "labels_by_pr.json"
    if not key.exists():
        pytest.skip(f"no answer key for {repo}")
    pulls = json.loads(key.read_text(encoding="utf-8"))["pull_requests"]
    return next(f"PR-{pr}" for pr, row in pulls.items() if row["source_case"] == source_case)


def case(repo: str, source_case: str = "") -> Case:
    """The first case, or the one published from `source_case`."""
    loaded = cases(repo)
    if not source_case:
        return loaded[0]
    wanted = case_id(repo, source_case)
    return next(c for c in loaded if c.case_id == wanted)
