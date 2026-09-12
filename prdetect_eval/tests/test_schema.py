"""The family table: both sides of a match resolve a type through it."""

import schema


def test_every_universal_name_resolves_to_a_family():
    """A type with no family resolves to "", which matches nothing: the family
    rung would be blind to exactly the names that reach beyond the corpora."""
    from detect import prompt
    missing = [name for name in prompt.types("halka", "review/v10-universal")
               if not schema.FAMILY_BY_TYPE.get(name)]
    assert missing == []


def test_demo_repos_coarse_names_share_the_catalogues_family_vocabulary():
    """demo_repo labels with `authz` where the catalogue says `missing_authz_check`.
    Both resolve through one table, so the two have to speak the same words or a
    fine-named report can never match a coarse label."""
    from detect import prompt
    families = {schema.FAMILY_BY_TYPE[name]
                for name in prompt.types("halka", "review/v10-universal")}
    coarse = {name: schema.FAMILY_BY_TYPE.get(name, "")
              for name in prompt.types("demo_repo", "review/v6")}
    unreachable = {name for name, family in coarse.items() if family not in families}
    # `business_logic` is the one kind the catalogue has no mechanical name for.
    assert unreachable == {"business_logic"}
