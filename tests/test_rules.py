"""The deterministic rules: each fires on a change of its shape, and on nothing else."""
from __future__ import annotations

from prdetect.cases import Case
from prdetect.detect import rules
from tests.repositories import NARROW, REPOSITORIES, WIDE, case, cases


def change(head: dict[str, str], added: dict[str, set[int]] | None = None, diff: str = "") -> Case:
    return Case(case_id="synthetic", pr_title="", pr_description="", changed_files=tuple(head), deleted_files=(),
                added_lines={name: frozenset(lines) for name, lines in (added or {}).items()},
                head_files=dict(head), context_files={}, diff=diff, is_defective=True)


def diff_of(path: str, removed: list[str], added: list[str], kept: list[str]) -> str:
    old, new = len(kept) + len(removed), len(kept) + len(added)
    body = [f" {line}" for line in kept] + [f"-{line}" for line in removed] + [f"+{line}" for line in added]
    return "\n".join([f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}",
                      f"@@ -1,{old} +1,{new} @@", *body, ""])


def test_a_bound_set_to_nothing_is_reported():
    found = rules.run(change({"app/defaults.py": "REPLAY_MAX_BATCH = 0\n"}, {"app/defaults.py": {1}}))
    assert [(f.type, f.rule) for f in found] == [("unsafe_default", "config.off-value")]


def test_a_bound_left_bounded_is_not():
    assert rules.run(change({"app/defaults.py": "REPLAY_MAX_BATCH = 500\n"}, {"app/defaults.py": {1}})) == []


def test_a_settings_field_set_to_nothing_is_reported():
    head = {"app/settings.py": "class Settings:\n    max_movement_rows: int = 0\n"}
    found = [f for f in rules.run(change(head, {"app/settings.py": {2}})) if f.rule == "config.off-value"]
    assert [(f.file, f.line) for f in found] == [("app/settings.py", 2)]


def test_a_list_written_as_one_string_is_reported():
    found = rules.run(change({"app/settings.py": 'RETRYABLE_STATUSES = "502,503,504"\n'}, {"app/settings.py": {1}}))
    assert [f.rule for f in found] == ["config.string-list"]


def test_a_test_left_asserting_less_is_reported():
    path, kept = "tests/test_worker.py", ["def test_worker():", "    result = run()"]
    diff = diff_of(path, removed=["    assert result.ok"], added=[], kept=kept)
    found = rules.run(change({path: "\n".join(kept) + "\n"}, diff=diff))
    assert [f.type for f in found] == ["missing_assertion"] and "assert" in found[0].quote


def test_a_rewritten_test_does_not_fire():
    path, kept = "tests/test_worker.py", ["def test_worker():", "    result = run()"]
    diff = diff_of(path, removed=["    assert result.ok"], added=['    assert result.status == "ok"'], kept=kept)
    head = "\n".join(kept + ['    assert result.status == "ok"']) + "\n"
    assert not [f for f in rules.run(change({path: head}, {path: {3}}, diff)) if f.rule == "test.weakened"]


def test_unittest_assertions_are_not_counted():
    assert rules.ASSERT.match("    assert x == 1")
    assert not rules.ASSERT.match("        self.assertEqual(x, 1)")
    path = "tests/test_report.py"
    kept = ["class ReportTests(TestCase):", "    def test_text(self):", "        text = render()"]
    diff = diff_of(path, removed=['        self.assertIn("GLUE250", text)'], added=[], kept=kept)
    assert rules.run(change({path: "\n".join(kept) + "\n"}, diff=diff)) == []


def test_a_local_variable_inside_a_function_is_not_a_setting():
    assert rules.ASSIGNMENT.match("    max_rows: int = 0")
    assert rules.ASSIGNMENT.match("MAX_ROWS = 0")
    assert not rules.ASSIGNMENT.match("        retries = 0")


def test_file_roles_are_read_off_the_file():
    assert rules.is_test_file("tests/test_worker.py") and rules.is_test_file("pkg/worker_test.py")
    assert rules.is_test_file("tests/conftest.py")
    assert not rules.is_test_file("tests/factories.py"), "a helper is not a test"
    assert rules.is_constants_module('"""doc"""\nimport os\nA = 1\nB = "x"\nC: int = 3\n')
    assert not rules.is_constants_module("A = 1\nB = 2\nC = 3\ndef f():\n    return A\n")
    assert not rules.is_constants_module("A = 1\n")


def test_no_rule_fires_on_a_clean_pull_request():
    for repo in REPOSITORIES:
        for item in cases(repo):
            if not item.is_defective:
                assert rules.run(item) == [], f"{repo}:{item.case_id}"


def test_a_secret_written_into_a_file_is_reported():
    found = rules.run(case(NARROW, "PR-14"))
    assert [(f.type, f.rule) for f in found] == [("hardcoded_credential", "secret.literal")]


def test_an_example_env_file_is_not_a_leak():
    assert rules.run(case(WIDE, "PR-107")) == []
