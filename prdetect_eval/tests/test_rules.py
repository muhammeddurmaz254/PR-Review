"""The deterministic rules for config and test files.

The positive cases they were written on lived in corpora that are gone; each is
kept here as the smallest change of the same shape.
"""

from corpora import NARROW, REPOSITORIES, WIDE, case, cases
from detect import rules
from schema import Case


def _change(head: dict[str, str], added: dict[str, set[int]] | None = None, diff: str = "") -> Case:
    return Case(case_id="synthetic", pair_id=None, variant="", difficulty="", primary_type=None,
                is_defective=True, pr_title="", pr_description="",
                changed_files=tuple(head), noise_files=(), deleted_files=(),
                added_lines={name: frozenset(lines) for name, lines in (added or {}).items()},
                head_files=dict(head), context_files={}, diff=diff, labels=(), distractors=())


def _test_diff(path: str, removed: list[str], added: list[str], kept: list[str]) -> str:
    old = len(kept) + len(removed)
    new = len(kept) + len(added)
    body = [f" {line}" for line in kept] + [f"-{line}" for line in removed] + [f"+{line}" for line in added]
    return "\n".join([f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}",
                      f"@@ -1,{old} +1,{new} @@", *body, ""])


def test_a_bound_set_to_nothing_is_reported():
    found = rules.run(_change({"app/defaults.py": "REPLAY_MAX_BATCH = 0\n"}, {"app/defaults.py": {1}}))
    assert [(f.type, f.rule) for f in found] == [("unsafe_default", "config.off-value")]
    assert found[0].file.endswith("defaults.py")


def test_a_bound_left_bounded_is_not():
    """The twin did not make the change, which is the whole separation."""
    assert rules.run(_change({"app/defaults.py": "REPLAY_MAX_BATCH = 500\n"}, {"app/defaults.py": {1}})) == []


def test_a_list_written_as_one_string_is_reported():
    found = rules.run(_change({"app/settings.py": 'RETRYABLE_STATUSES = "502,503,504"\n'},
                              {"app/settings.py": {1}}))
    assert [f.rule for f in found] == ["config.string-list"]


def test_a_test_left_asserting_less_is_reported():
    path = "tests/test_worker.py"
    kept = ["def test_worker():", "    result = run()"]
    diff = _test_diff(path, removed=["    assert result.ok"], added=[], kept=kept)
    found = rules.run(_change({path: "\n".join(kept) + "\n"}, diff=diff))
    assert [f.type for f in found] == ["missing_assertion"]
    assert "assert" in found[0].quote


def test_a_rewritten_test_does_not_fire():
    """One assertion out, one in, is a rewrite and not a weakening."""
    path = "tests/test_worker.py"
    kept = ["def test_worker():", "    result = run()"]
    diff = _test_diff(path, removed=["    assert result.ok"], added=['    assert result.status == "ok"'], kept=kept)
    head = "\n".join(kept + ['    assert result.status == "ok"']) + "\n"
    assert not [f for f in rules.run(_change({path: head}, {path: {3}}, diff)) if f.rule == "test.weakened"]


def test_a_settings_field_set_to_nothing_is_reported():
    """D29: a setting is also written as an annotated field on a settings
    object, and `max_movement_rows: int = 0` was invisible to a rule that knew
    only UPPER_CASE constants."""
    head = {"stock/settings.py": "class Settings:\n    max_movement_rows: int = 0\n"}
    found = [f for f in rules.run(_change(head, {"stock/settings.py": {2}})) if f.rule == "config.off-value"]
    assert [(f.file, f.line) for f in found] == [("stock/settings.py", 2)]


def test_no_rule_fires_on_a_clean_case():
    """The separation the whole approach rests on: a rule reads the change, and
    the clean twin did not make it."""
    for repo in REPOSITORIES:
        for item in cases(repo):
            if not item.is_defective:
                assert rules.run(item) == [], f"{repo}:{item.case_id}"


def test_the_config_and_test_rules_stay_quiet_on_both_repositories():
    """They were written on another corpus; neither repository draws anything from
    them, which is what `secret.literal` was added in D28 to change."""
    quiet = [rules.unbounded_default, rules.list_typed_as_text, rules.weakened_test]
    for repo in REPOSITORIES:
        for item in cases(repo):
            assert rules.run(item, quiet) == [], f"{repo}:{item.case_id}"


def test_a_secret_written_into_a_file_is_reported():
    found = rules.run(case(NARROW, "readme-compose-parolasi"))
    assert [(f.type, f.rule) for f in found] == [("hardcoded_credential", "secret.literal")]


def test_an_example_env_file_is_not_a_leak():
    """`.env.example` exists to show the shape of a secret; this pull request adds
    three of them and fired before this."""
    assert rules.run(case(WIDE, "temiz-11-ornek-ortam-dosyasi")) == []


def test_a_local_variable_inside_a_function_is_not_a_setting():
    """The indent cap: `retries = 0` in a function body is a counter."""
    assert rules.ASSIGNMENT.match("    max_rows: int = 0")
    assert rules.ASSIGNMENT.match("MAX_ROWS = 0")
    assert not rules.ASSIGNMENT.match("        retries = 0")


def test_unittest_assertions_are_not_counted():
    """D29: counting them cost a false alarm and bought nothing."""
    assert rules.ASSERT.match("    assert x == 1")
    assert not rules.ASSERT.match("        self.assertEqual(x, 1)")
    path = "tests/test_report.py"
    kept = ["class ReportTests(TestCase):", "    def test_text(self):", "        text = render()"]
    diff = _test_diff(path, removed=['        self.assertIn("GLUE250", text)'], added=[], kept=kept)
    assert rules.run(_change({path: "\n".join(kept) + "\n"}, diff=diff)) == []
