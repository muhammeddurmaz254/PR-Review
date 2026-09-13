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


def test_nothing_fires_on_the_other_corpora():
    """Measured: zero firings on 110 halka cases and 28 demo_repo cases."""
    for dataset in ("halka", "demo_repo"):
        for case in adapters.load_cases(DATASETS / f"{dataset}.eval.jsonl"):
            assert rules.run(case) == [], f"{dataset}:{case.case_id}"


def test_a_rewritten_test_does_not_fire():
    """One assertion out, one in, is a rewrite and not a weakening."""
    case = _case("zincir_dev", "clean-schema-v4")
    assert not [f for f in rules.run(case) if f.rule == "test.weakened"]
