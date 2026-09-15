"""Stage [3c]: the passages of a repository's docs that name what a change touches.

The corpus this was measured on is gone; the convention it quoted is kept here
as the smallest repository that writes one down.
"""

from corpora import NARROW, WIDE, cases
from detect import written_rules
from schema import Case

CONVENTIONS = """# Stock

Anything that takes stock away compares against `available()`, never against `on_hand`:
comparing against `on_hand` hands the same units to two orders.
"""
CHANGED = "def allocate(level, qty):\n    if level.on_hand >= qty:  # that nothing every\n        return True\n"


def _change(context: dict[str, str]) -> Case:
    path = "stock/allocate.py"
    return Case(case_id="synthetic", pair_id=None, variant="", difficulty="", primary_type=None,
                is_defective=True, pr_title="", pr_description="",
                changed_files=(path,), noise_files=(), deleted_files=(),
                added_lines={path: frozenset({2})}, head_files={path: CHANGED},
                context_files=context, diff="", labels=(), distractors=())


def test_the_convention_a_change_breaks_is_quoted():
    """The change compares against on_hand; the repository writes down why not."""
    passages = written_rules.collect(_change({"docs/conventions.md": CONVENTIONS}))
    assert any("available()" in p.text and "on_hand" in p.text for p in passages)


def test_a_repository_that_writes_nothing_gets_nothing():
    assert written_rules.collect(_change({})) == []
    for case in cases(NARROW):
        assert written_rules.collect(case) == []


def test_the_budget_is_a_ceiling():
    for case in cases(WIDE):
        assert sum(len(p.text) for p in written_rules.collect(case)) <= written_rules.BUDGET_CHARS


def test_only_code_names_on_changed_lines_count():
    """The first version read comments and prose too; the separation test caught
    it selecting passages by `that`, `nothing` and Turkish comment words."""
    names = written_rules.touched_names(_change({}))
    assert "on_hand" in names
    assert not names & {"self", "return", "None", "that", "nothing", "every"}


def test_a_passage_is_matched_only_where_it_names_code():
    text = "Any operation compares against `available()`, never against on_hand as a word."
    assert written_rules._named_in(text) == {"available"}


def test_the_section_says_most_changes_break_none():
    """The rule list that lost sent the model hunting; this must not read as a checklist."""
    passages = written_rules.collect(_change({"docs/conventions.md": CONVENTIONS}))
    rendered = "\n".join(written_rules.render(passages))
    assert "Most changes break none of them" in rendered
