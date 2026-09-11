"""Run layer 1 over a corpus: analyzer diagnostics each pull request introduces.

Every case's head revision and the corpus base are extracted with `git
archive` into a scratch directory, so the corpus repository is never checked
out or touched. The base is analysed once per distinct base commit.

    python run_static.py --dataset zincir --repo ../zincir_bench/repo \\
        --bin /path/to/venv/bin --run-id static-zincir
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adapters import eval_path, load_cases
from detect import static

HERE = Path(__file__).resolve().parent


def _extract(repo: Path, ref: str, into: str) -> None:
    subprocess.run(f"git -C '{repo}' archive '{ref}' | tar -x -C '{into}'", shell=True, check=True)


def _analyse(tree: str, tools: dict) -> list[static.Diagnostic]:
    return (static.run_ruff(tree, tools["ruff"])
            + static.run_pyright(tree, tools["pyright"], tools["python"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer 1: diagnostics a pull request introduces.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo", type=Path, required=True, help="the corpus git repository")
    parser.add_argument("--bin", type=Path, required=True, help="bin/ of the venv holding ruff and pyright")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    args = parser.parse_args(argv)

    tools = {"ruff": str(args.bin / "ruff"), "pyright": str(args.bin / "pyright"), "python": str(args.bin / "python")}
    record = {json.loads(l)["case_id"]: json.loads(l)
              for l in eval_path(args.dataset, HERE / "datasets").read_text().splitlines() if l.strip()}
    cases = load_cases(eval_path(args.dataset, HERE / "datasets"))[: args.limit]
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bases: dict[str, list[static.Diagnostic]] = {}
    rows = []
    for index, case in enumerate(cases, start=1):
        base_ref, head_ref = record[case.case_id]["base_commit"], record[case.case_id]["head_commit"]
        if base_ref not in bases:
            with tempfile.TemporaryDirectory() as tree:
                _extract(args.repo, base_ref, tree)
                bases[base_ref] = _analyse(tree, tools)
        with tempfile.TemporaryDirectory() as tree:
            _extract(args.repo, head_ref, tree)
            head = _analyse(tree, tools)
        new = static.introduced(bases[base_ref], head, case.added_lines)
        rows.append({"case_id": case.case_id, "is_defective": case.is_defective,
                     "diagnostics": [d.as_dict() for d in new]})
        print(f"[{index}/{len(cases)}] {case.case_id}: {len(new)} new", file=sys.stderr, flush=True)

    (run_dir / "static.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8", newline="\n")
    version = lambda cmd: subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "static", "dataset": args.dataset, "repo": str(args.repo),
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
        "ruff": version([tools["ruff"], "--version"]), "pyright": version([tools["pyright"], "--version"]),
        "ruff_select": static.RUFF_SELECT, "ruff_ignore": static.RUFF_IGNORE,
        "base_diagnostics": {k: len(v) for k, v in bases.items()},
        "introduced": sum(len(r["diagnostics"]) for r in rows),
        "python": platform.python_version(), "argv": list(argv if argv is not None else sys.argv[1:]),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"artefacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
