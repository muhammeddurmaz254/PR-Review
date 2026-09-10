"""Export zincir_bench into the harness's case format, one file per side.

zincir_bench is the second corpus, and it exists to answer three questions
halka_bench could not. Two of them decide how this exporter differs from
``build_halka.py``:

**A case can carry more than one defect.** Every halka case carries exactly
one, so "does the model stop after the first finding?" had no data behind it.
Here forty-three percent of the defective cases carry two or four, three of them
above the per-pull-request cap, and each finding is exported separately with its
own type, role and grounding. ``primary_type`` is the first of them and is not
the whole story any more -- anything reading it as "the case's type" is reading
a multi-defect case wrong.

**The split is part of the corpus, not of the run.** ``yaka`` is written at
build time from a fixed seed, by pair, and it lands in two files in two
directories. Nothing here chooses it, and nothing downstream may re-choose it:
halka is a development corpus today because two dozen decisions were taken while
looking at it, and no later split undoes that.

The third question -- when should the model speak in a test or a config file --
needs no special handling here beyond carrying ``file_role`` through, which the
loader now reads (section 0.2).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect import pack

HERE = Path(__file__).resolve().parent
# The in-project copy wins. Two copies of this corpus exist on this machine --
# the scratch directory it was first built in, and the one that lives beside the
# harness -- and with the scratch copy first every export silently came from
# outside the project: a fix applied to `PR-Review/zincir_bench` rebuilt its git
# branches, and the exported JSONL still carried the old commit.
CANDIDATES = (
    HERE.parents[1] / "zincir_bench",
    HERE.parents[2] / "zincir_repo_olusturma" / "zincir_bench",
)
SIDES = ("dev", "holdout")

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

ROLE_BY_NAME = {"application": "application", "test": "test", "config": "config"}


def find_bench(given: Path | None) -> Path:
    for candidate in ([given] if given else list(CANDIDATES)):
        if candidate and (candidate / "taxonomy.json").exists():
            return candidate
    raise SystemExit("zincir_bench not found; pass --bench")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def files_at(repo: Path, commit: str) -> dict[str, str]:
    """The whole tree at ``commit``.

    The repository is fifty files and under sixty kilobytes, so every case can
    carry all of it and still leave the window nearly empty. That is what makes
    the context question askable: run it with the changed files alone, run it
    again with the repository behind them, and the difference is what retrieval
    would have to buy.
    """
    names = [n for n in git(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if n]
    return {name: git(repo, "show", f"{commit}:{name}") for name in names}


def touched(diff: str) -> dict[str, list[int]]:
    """New-side line numbers the diff adds or rewrites, per file."""
    added: dict[str, set[int]] = {}
    filename = ""
    number = 0
    inside = False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            filename = line.split(" b/", 1)[-1].strip()
            inside = False
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
    return {name: sorted(lines) for name, lines in sorted(added.items())}


def finding(case_id: str, index: int, defect: dict, taxonomy: dict) -> dict:
    kind = defect["tur"]
    meta = taxonomy.get(kind, {})
    where = defect["konum"]
    # Only source reaches the model, so a label anywhere else is unreachable by
    # construction and must not be scored.
    in_scope = pack.is_code(where["dosya"])
    role = defect.get("file_role", "")
    if role and role not in ROLE_BY_NAME:
        raise SystemExit(f"{case_id}: unknown file_role {role!r}")
    return {
        "finding_id": defect.get("kusur_kimligi") or f"{case_id}-f{index}",
        "type": kind,
        "family": defect.get("aile") or meta.get("aile", ""),
        "in_scope": in_scope,
        "required": in_scope,
        # The label's role in its case. The role of the FILE is `file_role`,
        # and conflating the two is what section 0.2 separates.
        "role": "primary",
        "file_role": role,
        "file": where["dosya"],
        "start_line": where["baslangic_satiri"],
        "end_line": where["bitis_satiri"],
        "focus_start": where["baslangic_satiri"],
        "focus_end": where["bitis_satiri"],
        "anchor_rule": defect.get("kapsam_turu", ""),
        "anchor_text": "\n".join(where.get("capa_metni", [])),
        "title": defect.get("gerekce", ""),
        "in_diff": True,
        "cross_file": kind.startswith("crossfile_"),
        # A4b. The defect IS the removal: the line it removed is in no head
        # file, so the anchor is where the absence shows and `deleted_text`
        # says what is missing.
        "pure_deletion": bool(defect.get("silme_mi")),
        "deleted_text": defect.get("silinen_metin", ""),
        "deletion_consumer": defect.get("silineni_kullanan") or "",
        "equivalent_locations": [
            {"file": alt["dosya"], "start_line": alt["baslangic_satiri"],
             "end_line": alt["bitis_satiri"]}
            for alt in defect.get("esdeger_konumlar", [])
        ],
        "severity": "", "cwe": "",
        "rationale": defect.get("dayanak_notu", ""),
        "grounding": defect.get("dayanak", ""),
        "rule_ids": list(meta.get("kural_kimlikleri", [])),
        "product_scope": bool(meta.get("kural_kimlikleri")),
        "product_match_class": meta.get("eslesme_sinifi", ""),
    }


def build(bench: Path, out_dir: Path, keep_context: bool = True) -> dict[str, int]:
    taxonomy = json.loads((bench / "taxonomy.json").read_text(encoding="utf-8"))["turler"]
    repo = bench / "repo"
    written = {}
    for side in SIDES:
        source = bench / side / "corpus.jsonl"
        if not source.exists():
            raise SystemExit(f"{source} is missing. Build it with build/build_corpus.py")
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = case["vaka_kimligi"]
            if case.get("yaka") != side:
                raise SystemExit(f"{case_id}: filed under {side} but marked {case.get('yaka')!r}")
            head = case["head_commit"]
            changed = list(case["degisen_dosyalar"])
            tree = files_at(repo, head)
            missing = [name for name in changed if name not in tree]
            if missing:
                raise SystemExit(f"{case_id}: {missing} not in the tree at {head[:8]}")

            findings = [finding(case_id, index, defect, taxonomy)
                        for index, defect in enumerate(case.get("kusurlar", []), start=1)]
            rows.append({
                "case_id": case_id,
                "pair_id": case.get("cift_kimligi"),
                "variant": "buggy" if case["varyant"] == "kusurlu" else "clean",
                "difficulty": "hard",
                "primary_type": findings[0]["type"] if findings else None,
                "is_defective": bool(findings),
                "pr_title": case.get("baslik", ""),
                "pr_description": case.get("ozet", ""),
                "changed_files": changed,
                "noise_files": [], "deleted_files": [],
                "added_lines": touched(case["diff"]),
                "head_files": {name: tree[name] for name in changed},
                "context_files": ({name: text for name, text in tree.items()
                                   if name not in changed} if keep_context else {}),
                "diff": case["diff"],
                "findings": findings,
                "distractors": [],
                "base_commit": case.get("base_commit", ""),
                "head_commit": head,
                "branch": case.get("dal", ""),
                # Carried so a run cannot silently mix the two sides.
                "side": side,
                "expected_findings": case.get("beklenen_bulgu_sayisi", len(findings)),
            })
        out_path = out_dir / f"zincir_{side}.eval.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        written[side] = len(rows)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export zincir_bench, one file per side.")
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--no-context", action="store_true",
                        help="omit the unchanged files; the pack can then only be run at rung 0")
    args = parser.parse_args(argv)
    bench = find_bench(args.bench)
    written = build(bench, args.out, keep_context=not args.no_context)
    for side, count in written.items():
        print(f"Exported {count} {side} cases to {args.out / f'zincir_{side}.eval.jsonl'}")
    print("holdout is closed: see zincir_bench/README.md before running it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
