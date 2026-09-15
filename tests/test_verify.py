"""The verifier: its question, its answer schema, and the check on what it cites."""
from __future__ import annotations

import hashlib
import json

from prdetect.detect import agent, verify
from tests.test_agent import small_case

# Unpinned only on purpose, like the detector's prompt.
PINNED = "c4b371cb5599bf63"


def test_the_verifier_prompt_is_pinned():
    assert hashlib.sha256(verify.SYSTEM.encode("utf-8")).hexdigest().startswith(PINNED), "the verifier prompt changed"


def test_the_verifier_asks_for_the_failing_run_and_not_for_a_caller():
    assert "states in prose counts as evidence" in verify.SYSTEM
    assert "write the failure out" in verify.SYSTEM
    assert "never contradicts a claim on its own" in verify.SYSTEM
    assert verify.SYSTEM.endswith("however plausible it sounds.\n")


def test_the_verdict_is_generated_last_and_carries_no_negation():
    assert list(verify.SCHEMA["properties"]) == ["needed", "evidence", "reason", "verdict"]
    assert not any(v.startswith(("not", "no_", "non")) for v in verify.VERDICTS)


def test_an_unreadable_answer_is_unsettled():
    for text in ("", "not json", '{"verdict": "maybe"}'):
        assert verify.parse(text)["verdict"] == "unsettled"


def test_a_decisive_verdict_needs_evidence_that_exists():
    w = agent.Workspace(small_case())
    real = {"verdict": "established", "evidence": [{"file": "lib.py", "line": 1, "quote": "def charge(total):"}]}
    fake = {"verdict": "established", "evidence": [{"file": "lib.py", "line": 1, "quote": "def refund(x):"}]}
    assert verify.settle(real, w) == "established"
    assert verify.settle(fake, w) == "unsettled"
    assert verify.settle({"verdict": "contradicted", "evidence": []}, w) == "unsettled"


def test_a_deleted_line_can_be_cited_from_the_revision_before():
    assert verify.cited(agent.Workspace(small_case()), {"file": "app.py", "line": 2, "quote": "check(amount)"})


def test_a_broken_tool_call_is_resampled_once_and_a_repeat_is_not_run():
    calls = []

    def chat(messages, tools=None, schema=None, temperature=None, seed=None):
        calls.append(temperature)
        if schema is not None:
            return {"message": {"content": json.dumps({"needed": "", "evidence": [], "reason": "",
                                                       "verdict": "unsettled"})}}
        if len(calls) == 1:
            return {"error": "HTTPError 500: XML syntax error on line 4"}
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": "find_definition", "arguments": {"name": "charge"}}},
            {"function": {"name": "find_definition", "arguments": {"name": "charge"}}}]}}

    _, transcript, _ = agent.review(chat, verify.SYSTEM, "u", agent.Workspace(small_case()), verify.SCHEMA,
                                    max_calls=2, guide=verify.GUIDE, final_ask=verify.FINAL_ASK)
    assert calls[1] == 0.3, "resampled at a small temperature"
    assert any("resampled" in step for step in transcript)
    tools = [step for step in transcript if "tool" in step]
    assert "already made this exact call" in tools[1]["result"]


def test_the_named_claim_carries_its_kind_and_definition():
    claim = {"file": "a.py", "line": 3, "type": "off_by_one", "message": "Loop skips the last row"}
    text = verify.user_message(claim, "The index is one away from the range.", ["    3 + | x"])
    assert "`off_by_one` -- The index is one away" in text and "Loop skips the last row" in text


def test_the_located_claim_carries_its_sentence_and_not_its_name():
    claim = {"file": "app/quantity.py", "line": 72, "type": "off_by_one",
             "message": "Pallet list built by repeating the drop size"}
    text = verify.user_message_located(claim, "The index is one away from the range.", ["   72 + | x"])
    assert "Pallet list built by repeating the drop size" in text
    assert "off_by_one" not in text and "one away from the range" not in text
    assert "nothing wider and nothing narrower" in text


def test_a_claim_with_no_sentence_falls_back_to_its_name():
    claim = {"file": "a.py", "line": 3, "type": "off_by_one", "message": ""}
    assert verify.user_message_located(claim, "defn", ["3 | x"]) == verify.user_message(claim, "defn", ["3 | x"])


def test_only_claims_every_verifier_established_are_agreed():
    a = {"case_id": "c", "file": "a.py", "line": 1, "type": "x"}
    b = {"case_id": "c", "file": "a.py", "line": 9, "type": "y"}
    c = {"case_id": "c", "file": "b.py", "line": 3, "type": "z"}
    k = lambda x: (x["case_id"], x["file"], x["line"], x["type"])
    named = {k(a): {"verdict": "established"}, k(b): {"verdict": "established"}, k(c): {"verdict": "contradicted"}}
    located = {k(a): {"verdict": "established"}, k(b): {"verdict": "unsettled"}}
    assert verify.agreed([a, b, c], named, located) == [a]
    assert verify.agreed([a, b, c], named) == [a, b]
