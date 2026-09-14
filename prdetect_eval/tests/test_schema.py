"""The family table: both sides of a match resolve a type through it."""

from pathlib import Path

import pytest

import schema
from adapters import load_cases
from detect import prompt

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def test_every_catalogue_name_resolves_to_a_family():
    """A type with no family resolves to "", which matches nothing: the family
    rung would be blind to exactly the names that reach beyond the corpora."""
    missing = [name for name in prompt.types("halka") if not schema.FAMILY_BY_TYPE.get(name)]
    assert missing == []


def test_demo_repos_coarse_names_share_the_catalogues_family_vocabulary():
    """demo_repo labels with `authz` where the catalogue says `missing_authz_check`.
    Both resolve through one table, so the two have to speak the same words or a
    fine-named report can never match a coarse label."""
    path = DATASETS / "demo_repo.eval.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} is not built")
    families = {schema.FAMILY_BY_TYPE[name] for name in prompt.types("halka")}
    coarse = {label.type for case in load_cases(path) for label in case.labels}
    unreachable = {name for name in coarse if schema.FAMILY_BY_TYPE.get(name, "") not in families}
    # `business_logic` is the one kind the catalogue has no mechanical name for.
    assert unreachable == {"business_logic"}
