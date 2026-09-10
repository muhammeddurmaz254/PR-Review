"""Export halka_bench into the harness's case format.

Unlike the two corpora before it, halka_bench ships a real checkout: a small
multi-tenant Django service, one branch per case, and labels anchored to line
text rather than to line numbers. Three of its properties decide how this
exporter is written.

**The whole repository is small on purpose.** Fifty-five Python files, 1571
lines, about twelve thousand tokens -- so every file can be carried on every
case and still leave most of a 262k window free. That is what makes the
context question measurable here: the corpus can be run with the changed files
alone and again with the repository behind them, and the difference is the
value of retrieval, measured before any retrieval is built.

**Two thirds of the defects need that repository.** Thirty-three of the
forty-nine labels are grounded in a convention established elsewhere in the base
branch -- a predicate a sibling endpoint uses, a unit contract in
``common/money.py``, a docstring on a selector the pull request never opens.
Shipping only the diff would make them unreachable by construction.

**Every label is in scope.** ``taxonomy.json`` marks eight types outside
``birincil_kapsam``, but the reason is the *product's* rule-id vocabulary, not
the label: ``karsiliksiz`` means the existing static analyser has no id for the
type, and ``paylasimli`` means its id is shared so a match to the product is
ambiguous. Neither applies to a detector scored against the corpus's own ``tur``
field, which is unique. Excluding them dropped eleven real defects -- SSRF,
unsafe deserialization, a secret in a log, a missing lock -- from a measurement
that has nothing to do with the product's vocabulary. The corpus's own
classification is carried through on each finding so the product-facing view
stays reconstructible.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect import pack
from typing import Sequence

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "halka_bench"

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def files_at(repo: Path, commit: str) -> dict[str, str]:
    """The whole tree at ``commit``.

    No extension filter. The repository is fifty-nine text files totalling 52 KB,
    and the non-Python ones carry contracts the labels rest on --
    ``docs/conventions.md`` states the sixteen conventions, and two cases turn on
    a requirements file.
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


def build(bench: Path, out_path: Path, keep_context: bool = True) -> int:
    taxonomy = json.loads((bench / "taxonomy.json").read_text(encoding="utf-8"))["turler"]
    repo = bench / "repo"
    rows = []
    for line in (bench / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        case_id = case["vaka_kimligi"]
        head = case["head_commit"]
        changed = list(case["degisen_dosyalar"])
        tree = files_at(repo, head)
        missing = [name for name in changed if name not in tree]
        if missing:
            raise SystemExit(f"{case_id}: {missing} not in the tree at {head[:8]}")

        findings = []
        for index, defect in enumerate(case.get("kusurlar", []), start=1):
            kind = defect["tur"]
            meta = taxonomy.get(kind, {})
            where = defect["konum"]
            # Only code is put in front of the model, so a label in a manifest or
            # a document is unreachable by construction and must not be scored.
            in_scope = pack.is_code(where["dosya"])
            findings.append({
                "finding_id": f"{case_id}-f{index}",
                "type": kind,
                "family": case.get("aile", ""),
                "in_scope": in_scope,
                "required": in_scope,
                "role": "primary",
                "file": where["dosya"],
                "start_line": where["baslangic_satiri"],
                "end_line": where["bitis_satiri"],
                # The corpus anchors a defect to the exact lines it lives on, so
                # region and focus are the same span; there is no wider
                # "function that contains it" to fall back on.
                "focus_start": where["baslangic_satiri"],
                "focus_end": where["bitis_satiri"],
                "anchor_rule": defect.get("kapsam_turu", ""),
                # The corpus's product-facing classification, kept but not used
                # for scope: `birincil_kapsam` and `eslesme_sinifi` describe what
                # the existing analyser can express, not what a defect is.
                "product_scope": bool(meta.get("birincil_kapsam")),
                "product_match_class": defect.get("eslesme_sinifi", ""),
                "rule_ids": defect.get("kural_kimlikleri", []),
                "anchor_text": "\n".join(where.get("capa_metni", [])),
                "title": defect.get("gerekce", ""),
                "in_diff": True,
                "cross_file": kind.startswith("crossfile_"),
                "pure_deletion": False,
                "equivalent_locations": [
                    {"file": alt["dosya"], "start_line": alt["baslangic_satiri"],
                     "end_line": alt["bitis_satiri"]}
                    for alt in defect.get("esdeger_konumlar", [])
                ],
                "severity": "", "cwe": "",
                "rationale": defect.get("dayanak_notu", ""),
                # The corpus's own statement that the evidence is visible, and
                # where. It is the reason a label was allowed in at all.
                "grounding": defect.get("dayanak", ""),
            })

        rows.append({
            "case_id": case_id,
            "pair_id": case.get("cift_kimligi"),
            "variant": "buggy" if case["varyant"] == "kusurlu" else "clean",
            "difficulty": "hard",
            "primary_type": findings[0]["type"] if findings else None,
            "is_defective": bool(case.get("kusurlar")),
            "pr_title": case.get("baslik", ""),
            "pr_description": case.get("ozet", ""),
            "changed_files": changed,
            "noise_files": [], "deleted_files": [],
            "added_lines": touched(case["diff"]),
            "head_files": {name: tree[name] for name in changed},
            "context_files": ({name: text for name, text in tree.items() if name not in changed}
                              if keep_context else {}),
            "diff": case["diff"],
            "findings": findings,
            "distractors": [],
            "base_commit": case.get("base_commit", ""),
            "head_commit": head,
            "branch": case.get("dal", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export halka_bench into the harness case format.")
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--out", type=Path, default=HERE / "halka.eval.jsonl")
    parser.add_argument("--no-context", action="store_true",
                        help="omit the unchanged files; the pack can then only be run at rung 0")
    args = parser.parse_args(argv)
    count = build(args.bench, args.out, keep_context=not args.no_context)
    print(f"Exported {count} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
