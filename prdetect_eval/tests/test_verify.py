"""The verifier agent: layer 3 checks what it cites, and the orchestrator is robust."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect import agent, verify
from tests.test_agent import _case


def test_a_decisive_verdict_needs_evidence_that_exists():
    w = agent.Workspace(_case())
    real = {"verdict": "established", "evidence": [{"file": "lib.py", "line": 1, "quote": "def charge(total):"}]}
    fake = {"verdict": "established", "evidence": [{"file": "lib.py", "line": 1, "quote": "def refund(x):"}]}
    none = {"verdict": "contradicted", "evidence": []}
    assert verify.settle(real, w) == "established"
    assert verify.settle(fake, w) == "unsettled"
    assert verify.settle(none, w) == "unsettled"


def test_a_deleted_line_can_be_cited_from_the_revision_before():
    w = agent.Workspace(_case())
    assert verify.cited(w, {"file": "app.py", "line": 2, "quote": "check(amount)"})


def test_the_two_policies():
    assert verify.keeps("established", "strict") and not verify.keeps("unsettled", "strict")
    assert verify.keeps("unsettled", "lenient") and not verify.keeps("contradicted", "lenient")


def test_the_verdict_is_generated_last_and_carries_no_negation():
    assert list(verify.SCHEMA["properties"]) == ["needed", "evidence", "reason", "verdict"]
    assert not any(v.startswith(("not", "no_", "non")) for v in verify.VERDICTS)


def test_a_broken_tool_call_is_resampled_once_and_a_repeat_is_not_executed():
    calls = []
    def chat(messages, tools=None, schema=None, temperature=None, seed=None):
        calls.append(temperature)
        if schema is not None:
            return {"message": {"content": json.dumps({"needed": "", "evidence": [], "reason": "", "verdict": "unsettled"})}}
        if len(calls) == 1:
            return {"error": "HTTPError 500: XML syntax error on line 4"}
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": "find_definition", "arguments": {"name": "charge"}}},
            {"function": {"name": "find_definition", "arguments": {"name": "charge"}}}]}}
    text, transcript, n = agent.review(chat, "s", "u", agent.Workspace(_case()), verify.SCHEMA, max_calls=2,
                                       guide=verify.GUIDE, final_ask=verify.FINAL_ASK)
    assert calls[1] == 0.3, "resampled at a small temperature"
    assert any("resampled" in s for s in transcript)
    tools = [s for s in transcript if "tool" in s]
    assert "already made this exact call" in tools[1]["result"]


def test_a_stopped_verification_resumes_from_what_it_wrote(tmp_path):
    """A corpus pass is long and the tunnel drops; the verdicts already on disk are the run."""
    import run_verify
    rows = [{"case_id": "c", "file": "a.py", "line": 3, "type": "x", "verdict": "established"}]
    (tmp_path / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    stored = {(r["case_id"], r["file"], r["line"], r["type"]): r
              for r in map(json.loads, (tmp_path / "verdicts.jsonl").read_text().splitlines())}
    assert ("c", "a.py", 3, "x") in stored
    assert "--resume" in Path(run_verify.__file__).read_text(encoding="utf-8")


def test_the_measured_verifier_question_is_pinned():
    """It has a published score now; a silent edit would invalidate it."""
    import hashlib
    assert hashlib.sha256(verify.SYSTEM.encode("utf-8")).hexdigest().startswith("7412248db7b354bb")


def test_the_contract_variant_adds_one_clause_and_nothing_else():
    assert verify.SYSTEM_CONTRACT.replace(verify.CONTRACT_CLAUSE, "", 1) == verify.SYSTEM
    assert "states in prose" in verify.CONTRACT_CLAUSE
    assert "Quote the prose the way you quote code." in verify.SYSTEM_CONTRACT


def test_the_default_verify_stage_is_strict_with_the_contract_clause(tmp_path):
    """Made the default after three corpora: precision up on all of them."""
    import run_verify
    parser_source = Path(run_verify.__file__).read_text(encoding="utf-8")
    assert '"--policy"' in parser_source and 'default="strict"' in parser_source
    assert "BooleanOptionalAction, default=True" in parser_source
    assert "run_dir / \"predictions.jsonl\"" in parser_source, "the chosen policy is the run's own output"


# --- The precedent moved from the catalogue to this stage --------------------

def test_precedent_prompt_keeps_the_contract_clause_and_the_unsettled_rule():
    system = verify.system_for(contract=False, precedent=True)
    assert system == verify.SYSTEM_PRECEDENT
    assert "states in prose counts as evidence" in system
    assert system.rstrip().endswith("however plausible it sounds.")
    assert "the revision before this change" in system


def test_precedent_is_off_unless_asked_for():
    assert verify.system_for(contract=True, precedent=False) == verify.SYSTEM_CONTRACT
    assert verify.system_for(contract=False, precedent=False) == verify.SYSTEM
    assert "missing there is missing everywhere" not in verify.SYSTEM_CONTRACT


def test_measured_verifier_prompts_are_unchanged():
    """Same rule as the detector's: a text a number was measured under is
    pinned, and unpinned only by re-measuring."""
    import hashlib
    pinned = {"SYSTEM": "7412248db7b354bb",
              "SYSTEM_CONTRACT": "975e1ee003b0cce7"}
    for name, digest in pinned.items():
        actual = hashlib.sha256(getattr(verify, name).encode("utf-8")).hexdigest()
        assert actual.startswith(digest), f"{name} changed; re-measure or revert"


def test_mechanism_asks_nothing_of_the_rest_of_the_repository():
    """The point of it: a repository that locks nothing anywhere still loses
    money to a double spend, so the verdict cannot depend on what the
    neighbours do."""
    system = verify.system_for(contract=True, mechanism=True)
    assert system == verify.SYSTEM_MECHANISM
    assert "write the failure out" in system
    assert "even where the rest of the repository does the same thing" in system
    assert "comparable place" not in system


def test_the_two_clauses_are_not_combined():
    import pytest
    with pytest.raises(ValueError):
        verify.system_for(contract=True, precedent=True, mechanism=True)
    assert verify.system_for(contract=True) == verify.SYSTEM_CONTRACT


def test_the_callers_note_removes_the_argument_that_killed_true_findings():
    """Three of four true findings the mechanism clause contradicted fell to
    'nothing in the repository calls it'."""
    system = verify.system_for(contract=True, mechanism=True, callers=True)
    assert system == verify.SYSTEM_MECHANISM_CALLERS
    assert "never contradicts a claim on its own" in system
    assert "write the failure out" in system          # the mechanism clause stays
    assert "Whether anything in this repository calls" not in verify.SYSTEM_MECHANISM


def test_the_callers_note_needs_the_mechanism_clause():
    assert verify.system_for(contract=True, callers=True) == verify.SYSTEM_CONTRACT
