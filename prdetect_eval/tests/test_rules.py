"""The deterministic rules for config and test files."""

import pathlib

import adapters
from detect import rules

DATASETS = pathlib.Path(__file__).resolve().parent.parent / "datasets"


def _case(dataset, case_id):
    cases = adapters.load_cases(DATASETS / f"{dataset}.eval.jsonl")
    return next(c for c in cases if c.case_id == case_id)


def test_a_bound_set_to_nothing_is_reported():
    found = rules.run(_case("zincir_dev", "replay-01-kusurlu"))
    assert [(f.type, f.rule) for f in found] == [("unsafe_default", "config.off-value")]
    assert found[0].file.endswith("defaults.py")


def test_a_list_written_as_one_string_is_reported():
    found = rules.run(_case("zincir_dev", "retryable-01-kusurlu"))
    assert [f.rule for f in found] == ["config.string-list"]


def test_a_test_left_asserting_less_is_reported():
    found = rules.run(_case("zincir_dev", "ckpt-01-kusurlu"))
    assert [f.type for f in found] == ["missing_assertion"]
    assert "assert" in found[0].quote


def test_the_clean_twins_stay_silent():
    """The twin did not make the change, which is the whole separation."""
    for case_id in ("clean-schema-v4", "clean-inventory-handler"):
        assert rules.run(_case("zincir_dev", case_id)) == []


def test_no_rule_fires_on_a_clean_case():
    """The separation the whole approach rests on: a rule reads the change, and
    the clean twin did not make it. Measured across all three corpora."""
    for dataset in ("halka", "demo_repo", "zincir_dev"):
        for case in adapters.load_cases(DATASETS / f"{dataset}.eval.jsonl"):
            if not case.is_defective:
                assert rules.run(case) == [], f"{dataset}:{case.case_id}"


def test_the_config_and_test_rules_stay_quiet_outside_zincir():
    """They were written on zincir_bench; halka and demo_repo draw nothing from
    them, which is what `secret.literal` was added in D28 to change."""
    quiet = [rules.unbounded_default, rules.list_typed_as_text, rules.weakened_test]
    for dataset in ("halka", "demo_repo"):
        for case in adapters.load_cases(DATASETS / f"{dataset}.eval.jsonl"):
            assert rules.run(case, quiet) == [], f"{dataset}:{case.case_id}"


def test_a_secret_written_into_a_file_is_reported():
    found = rules.run(_case("demo_repo", "readme-compose-parolasi"))
    assert [(f.type, f.rule) for f in found] == [("hardcoded_credential", "secret.literal")]


def test_an_example_env_file_is_not_a_leak():
    """`.env.example` exists to show the shape of a secret; halka's
    `temiz-11-ornek-ortam-dosyasi` is three of them and fired before this."""
    assert rules.run(_case("halka", "temiz-11-ornek-ortam-dosyasi")) == []


def test_a_rewritten_test_does_not_fire():
    """One assertion out, one in, is a rewrite and not a weakening."""
    case = _case("zincir_dev", "clean-schema-v4")
    assert not [f for f in rules.run(case) if f.rule == "test.weakened"]
