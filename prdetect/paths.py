"""Where the project reads and writes, all under the project root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
DATA = ROOT / "data"
CASES = DATA / "cases"
ANSWER_KEYS = DATA / "answer_keys"
BITBUCKET_CACHE = DATA / "cache" / "bitbucket"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


def cases_file(repo: str) -> Path:
    """The pull requests fetched from a repository, one case per line."""
    return CASES / f"{repo}.jsonl"


def answer_key_file(repo: str) -> Path:
    """A repository's labels, keyed by pull request id. Never shown to the model."""
    return ANSWER_KEYS / f"{repo}.json"
