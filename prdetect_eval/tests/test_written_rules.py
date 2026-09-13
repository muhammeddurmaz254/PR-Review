"""Stage [3c]: the passages of a repository's docs that name what a change touches."""

import pathlib

import adapters
from detect import written_rules

DATASETS = pathlib.Path(__file__).resolve().parent.parent / "datasets"


def _case(dataset, case_id):
    return next(c for c in adapters.load_cases(DATASETS / f"{dataset}.eval.jsonl") if c.case_id == case_id)


def test_the_convention_a_change_breaks_is_quoted():
    """stock-01 compares against on_hand; the repository writes down why not."""
    passages = written_rules.collect(_case("stock_dev", "stock-01-defective"))
    assert any("available()" in p.text and "on_hand" in p.text for p in passages)


def test_a_repository_that_writes_nothing_gets_nothing():
    for case in adapters.load_cases(DATASETS / "demo_repo.eval.jsonl"):
        assert written_rules.collect(case) == []


def test_the_budget_is_a_ceiling():
    for case in adapters.load_cases(DATASETS / "stock_dev.eval.jsonl"):
        assert sum(len(p.text) for p in written_rules.collect(case)) <= written_rules.BUDGET_CHARS


def test_only_code_names_on_changed_lines_count():
    """The first version read comments and prose too; the separation test caught
    it selecting passages by `that`, `nothing` and Turkish comment words."""
    names = written_rules.touched_names(_case("stock_dev", "stock-01-defective"))
    assert "on_hand" in names
    assert not names & {"self", "return", "None", "that", "nothing", "every"}


def test_a_passage_is_matched_only_where_it_names_code():
    text = "Any operation compares against `available()`, never against on_hand as a word."
    assert written_rules._named_in(text) == {"available"}


def test_the_section_says_most_changes_break_none():
    """The rule list that lost sent the model hunting; this must not read as a checklist."""
    rendered = "\n".join(written_rules.render(written_rules.collect(_case("stock_dev", "stock-01-defective"))))
    assert "Most changes break none of them" in rendered
