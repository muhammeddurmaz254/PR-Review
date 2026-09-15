"""The verifier's tools, the rebuilt revision before the change, and the tool loop."""
from __future__ import annotations

import json

from prdetect.cases import Case
from prdetect.detect import agent, contract

HEAD = {"app.py": "def pay(amount):\n    return charge(amount)\n",
        "lib.py": "def charge(total):\n    return total\n\nLIMIT = 3\n"}
DIFF = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,3 +1,2 @@
 def pay(amount):
-    check(amount)
     return charge(amount)
"""


def small_case(head=HEAD, diff=DIFF) -> Case:
    return Case(case_id="c", pr_title="", pr_description="", changed_files=("app.py",), deleted_files=(),
                added_lines={"app.py": frozenset({2})}, head_files={"app.py": head["app.py"]},
                context_files={k: v for k, v in head.items() if k != "app.py"}, diff=diff)


def test_the_base_revision_is_the_diff_reversed_onto_head():
    base = agent.base_revision(small_case())
    assert base["app.py"].split("\n")[:3] == ["def pay(amount):", "    check(amount)", "    return charge(amount)"]
    assert base["lib.py"] == HEAD["lib.py"]


def test_the_tools_read_the_repository_not_only_the_change():
    w = agent.Workspace(small_case())
    assert "lib.py:1: def charge(total):" in w.find_definition("charge")
    assert "app.py:2" in w.find_usages("charge")
    assert "lib.py:4" in w.find_definition("LIMIT")
    assert "    2 + |     return charge(amount)" in w.read_file("app.py")
    assert "check(amount)" in w.read_base("app.py")
    assert "no such file" in w.read_file("nope.py")
    assert w.search("zzz") == "(nothing found)"
    assert w.call("rm_rf", {}) == "unknown tool 'rm_rf'"


def test_the_loop_looks_then_answers_under_the_schema():
    script = [
        {"message": {"content": "", "tool_calls": [{"function": {"name": "find_definition",
                                                                  "arguments": {"name": "charge"}}}]}},
        {"message": {"content": "done"}},
        {"message": {"content": json.dumps({"findings": [
            {"file": "app.py", "line": 2, "type": "x", "title": "t", "confidence": 0.9}]})}},
    ]
    seen = []

    def chat(messages, tools=None, schema=None):
        seen.append((bool(tools), schema is not None))
        return script[len(seen) - 1]

    text, transcript, calls = agent.review(chat, "sys", "user", agent.Workspace(small_case()), {"type": "object"},
                                           max_calls=4, guide="", final_ask="answer")
    assert calls == 1 and transcript[0]["tool"] == "find_definition"
    assert seen[-1] == (False, True), "the answer is asked for under the schema, without tools"
    assert [(r.file, r.line) for r in contract.parse(text)[0]] == [("app.py", 2)]


def test_the_budget_stops_a_reviewer_that_never_stops_looking():
    forever = {"message": {"content": "", "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}]}}
    final = {"message": {"content": '{"findings": []}'}}

    def chat(messages, tools=None, schema=None):
        return final if schema is not None else forever

    _, _, calls = agent.review(chat, "s", "u", agent.Workspace(small_case()), {"type": "object"},
                               max_calls=3, guide="", final_ask="answer")
    assert calls == 3
