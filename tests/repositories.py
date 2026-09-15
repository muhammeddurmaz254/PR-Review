"""The two labelled repositories the tests read, as fetched from Bitbucket.

    python -m prdetect.cli.fetch --repo dar_olcum_reposu
    python -m prdetect.cli.fetch --repo genis_olcum_reposu

A test that needs one skips, naming that command, when it has not been fetched.
"""
from __future__ import annotations

import pytest

from prdetect import paths
from prdetect.cases import Case, load_cases

NARROW = "dar_olcum_reposu"
WIDE = "genis_olcum_reposu"
REPOSITORIES = (NARROW, WIDE)

_loaded: dict[str, list[Case]] = {}


def cases(repo: str) -> list[Case]:
    path = paths.cases_file(repo)
    if not path.exists():
        pytest.skip(f"{path.name} is not fetched: python -m prdetect.cli.fetch --repo {repo}")
    if repo not in _loaded:
        _loaded[repo] = load_cases(path)
    return _loaded[repo]


def case(repo: str, case_id: str | None = None) -> Case:
    """The first pull request, or the one named."""
    loaded = cases(repo)
    if case_id is None:
        return loaded[0]
    return next(item for item in loaded if item.case_id == case_id)
