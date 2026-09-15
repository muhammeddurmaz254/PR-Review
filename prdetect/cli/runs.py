"""What the stages share about a run directory: its manifest, its rows, its cases."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from prdetect import paths
from prdetect.cases import Case, load_cases


def commit() -> str:
    """The project's git commit, so a run records the code that made it."""
    result = subprocess.run(("git", "rev-parse", "--short", "HEAD"), cwd=paths.ROOT,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance(argv: Sequence[str] | None) -> dict:
    return {"created_utc": now(), "commit": commit(), "python": platform.python_version(),
            "argv": list(argv if argv is not None else sys.argv[1:])}


def read_rows(path: Path) -> list[dict]:
    """A JSON-lines file as rows; empty when the file does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        raise SystemExit(f"{run_dir} is not a run: it has no config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8", newline="\n")


def cases_file(manifest: dict) -> Path:
    """The cases file a run was made against.

    The manifest records the file the run actually read, which may have been
    given by path; the repository's default file is used only when that one is gone.
    """
    recorded = manifest.get("cases_file")
    if recorded and Path(recorded).exists():
        return Path(recorded)
    return paths.cases_file(manifest["repository"])


def cases_by_id(manifest: dict) -> dict[str, Case]:
    return {case.case_id: case for case in load_cases(cases_file(manifest))}


def claims_file(run_dir: Path) -> Path:
    """The claims a later stage reads from a detect or continue run: the ones the anchor check let through.

    The anchor check also corrects line numbers, so reading the file before it
    would carry the uncorrected lines forward.
    """
    anchored = run_dir / "predictions.anchored.jsonl"
    return anchored if anchored.exists() else run_dir / "predictions.jsonl"


def run_dir(run: str, out: Path = paths.RUNS) -> Path:
    """A run given by id under `out`, or by path."""
    given = Path(run)
    return given if given.is_dir() else out / run
