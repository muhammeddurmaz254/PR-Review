"""The stages from the command line, without a model."""
from __future__ import annotations

import json

import pytest

from prdetect.cli import continue_review, detect, publish, report, report_txt, rules, runs
from tests.repositories import NARROW, WIDE, cases
from tests.test_ollama import DyingOllama, serve


def test_resume_refuses_to_mix_two_prompts(tmp_path):
    cases(WIDE)
    run_dir = tmp_path / "half"
    run_dir.mkdir()
    (run_dir / "system_prompt.txt").write_text("a prompt this run was not made under\n")
    (run_dir / "responses.jsonl").write_text(json.dumps({"case_id": "PR-1", "text": '{"findings": []}'}) + "\n")
    with pytest.raises(SystemExit, match="different system prompt"):
        detect.main(["--repo", WIDE, "--stub", "silent", "--resume", "--run-id", "half",
                     "--out", str(tmp_path), "--quiet"])


def test_resume_reuses_the_answers_already_on_disk(tmp_path):
    cases(WIDE)
    args = ["--repo", WIDE, "--stub", "silent", "--limit", "4", "--run-id", "part", "--out", str(tmp_path), "--quiet"]
    assert detect.main(args) == 0
    first = (tmp_path / "part" / "responses.jsonl").read_text().splitlines()
    assert len(first) == 4
    (tmp_path / "part" / "responses.jsonl").write_text("\n".join(first[:2]) + "\n")
    assert detect.main(args + ["--resume"]) == 0
    assert runs.read_manifest(tmp_path / "part")["reused_answers"] == 2
    assert len((tmp_path / "part" / "responses.jsonl").read_text().splitlines()) == 4


def test_a_run_that_lost_its_server_does_not_look_finished(tmp_path):
    cases(WIDE)
    server, url = serve(DyingOllama)
    code = detect.main(["--repo", WIDE, "--model", "test-model", "--base-url", url, "--timeout", "2",
                        "--limit", "2", "--run-id", "dead", "--out", str(tmp_path), "--quiet"])
    server.shutdown()
    manifest = runs.read_manifest(tmp_path / "dead")
    assert code == 1 and manifest["complete"] is False and manifest["call_failures"] == 2


def test_a_dry_run_writes_the_prompts_and_nothing_else(tmp_path):
    loaded = cases(NARROW)
    assert detect.main(["--repo", NARROW, "--dry-run", "--run-id", "dry", "--out", str(tmp_path), "--quiet"]) == 0
    packs = runs.read_rows(tmp_path / "dry" / "packs.jsonl")
    assert [row["case_id"] for row in packs] == [item.case_id for item in loaded]
    assert runs.read_rows(tmp_path / "dry" / "predictions.jsonl") == []
    assert runs.read_manifest(tmp_path / "dry")["complete"] is True


def test_the_stages_chain_into_a_report(tmp_path):
    loaded = cases(NARROW)
    out = str(tmp_path)
    assert detect.main(["--repo", NARROW, "--stub", "silent", "--run-id", "d", "--out", out, "--quiet"]) == 0

    # One claim on one file of a two-file pull request, as the detector would write it.
    two = next(item for item in loaded if sum(name.endswith(".py") for name in item.head_files) >= 2)
    first = sorted(name for name in two.head_files if name.endswith(".py"))[0]
    claim = {"case_id": two.case_id, "file": first, "line": 1, "end_line": 1, "type": "xss", "confidence": 0.9,
             "detector": "review.llm", "stage": "detect", "message": "a claim"}
    runs.write_rows(tmp_path / "d" / "predictions.jsonl", [claim])
    runs.write_rows(tmp_path / "d" / "predictions.anchored.jsonl", [claim])

    assert continue_review.main(["--run", "d", "--dry-run", "--out", out, "--quiet"]) == 0
    trace = runs.read_rows(tmp_path / "d-cont" / "continuation.jsonl")
    assert [row["case_id"] for row in trace] == [two.case_id]
    assert "# STILL TO REVIEW" in trace[0]["user"] and f"`{first}`:1 xss" in trace[0]["user"]

    assert rules.main(["--repo", NARROW, "--run-id", "r", "--out", out, "--quiet"]) == 0
    assert publish.main(["--run", "d-cont", "--rules", "r", "--run-id", "p", "--out", out, "--quiet"]) == 0
    published = runs.read_rows(tmp_path / "p" / "predictions.jsonl")
    assert {row["case_id"] for row in published} == {two.case_id, "PR-14"}

    built = report.build(tmp_path / "p")
    assert built["repository"] == NARROW and built["summary"]["findings"] == 2
    assert set(built["summary"]["scores"]) == {"location", "family", "type"}
    text = report_txt.render(built)
    assert "PR #14" in text and f"PR #{two.pull_request['id']}" in text
