"""Stage [4b]: review what the first read skipped, and nothing else."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters import load_cases
from detect import contract, continuation, pack

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def _multi_file_case():
    for case in load_cases(DATASETS / "halka.eval.jsonl"):
        code = sorted(name for name in case.head_files if pack.is_code(name))
        if len(code) >= 2:
            return case, code
    raise AssertionError("halka has a pull request touching two code files")


def test_a_silent_first_read_is_not_reopened():
    """Silence is the speak-or-stay-silent decision; this stage does not touch it."""
    case, _ = _multi_file_case()
    assert continuation.targets(case, []) == []


def test_only_the_unreported_code_files_are_asked_about():
    case, code = _multi_file_case()
    remaining = continuation.targets(case, [code[0]])
    assert remaining == code[1:]
    assert all(pack.is_code(name) for name in remaining)
    assert continuation.targets(case, code) == []


def test_the_trailer_records_what_is_done_and_names_only_what_is_left():
    case, code = _multi_file_case()
    done = [{"file": code[0], "line": 3, "type": "x", "message": "m"}]
    text = continuation.trailer(done, code[1:])
    head, _, rest = text.partition("# STILL TO REVIEW")
    assert f"`{code[0]}`:3" in head
    assert f"`{code[0]}`" not in rest
    assert all(f"`{name}`" in rest for name in code[1:])
    # No prior is added: the system prompt's own guidance on silence stands.
    assert "empty `findings` list" in text
    assert "probably" not in text and "likely" not in text


def test_a_report_outside_the_files_asked_about_is_dropped():
    reports = [contract.Report("a.py", 1, "x", "t", 0.9), contract.Report("b.py", 2, "x", "t", 0.9)]
    assert [r.file for r in continuation.within(reports, ["b.py"])] == ["b.py"]


def test_the_verify_stages_read_the_claims_the_quote_gate_let_through(tmp_path):
    """The gate's corrections must reach the chain: `ckpt-01-kusurlu` was scored
    at line 60 after the gate had moved it to 61, where the corpus anchors it."""
    from adapters import claims_path
    (tmp_path / "predictions.jsonl").write_text("", encoding="utf-8")
    assert claims_path(tmp_path).name == "predictions.jsonl"
    (tmp_path / "predictions.anchored.jsonl").write_text("", encoding="utf-8")
    assert claims_path(tmp_path).name == "predictions.anchored.jsonl"
    # Both readers go through the one function, so they cannot disagree.
    import run_challenge, run_regate
    assert run_challenge.claims_path is claims_path and run_regate.claims_path is claims_path
