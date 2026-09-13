"""One finding per code site."""

from detect import dedupe


def claim(case_id, file, line, confidence, type_="x"):
    return {"case_id": case_id, "file": file, "line": line,
            "confidence": confidence, "type": type_}


def test_two_claims_on_one_site_become_the_more_confident_one():
    claims = [claim("c", "a.py", 53, 0.7, "path_traversal"),
              claim("c", "a.py", 57, 0.9, "command_injection")]
    kept = dedupe.one_per_site(claims)
    assert [c["type"] for c in kept] == ["command_injection"]


def test_a_second_finding_far_enough_away_is_kept():
    claims = [claim("c", "a.py", 10, 0.9), claim("c", "a.py", 40, 0.8)]
    assert len(dedupe.one_per_site(claims)) == 2


def test_the_same_line_in_another_file_is_another_site():
    claims = [claim("c", "a.py", 10, 0.9), claim("c", "b.py", 10, 0.8)]
    assert len(dedupe.one_per_site(claims)) == 2


def test_cases_do_not_shadow_each_other():
    claims = [claim("one", "a.py", 10, 0.9), claim("two", "a.py", 11, 0.8)]
    assert len(dedupe.one_per_site(claims)) == 2


def test_order_is_the_order_it_was_given():
    """The file is a record of the run, not a ranking."""
    claims = [claim("c", "a.py", 10, 0.5), claim("c", "b.py", 10, 0.9),
              claim("c", "b.py", 12, 0.4)]
    kept = dedupe.one_per_site(claims)
    assert [(c["file"], c["line"]) for c in kept] == [("a.py", 10), ("b.py", 10)]


def test_a_claim_with_no_line_is_never_merged_away():
    """An unanchored claim has no site, so it cannot be a duplicate of one."""
    claims = [claim("c", "a.py", None, 0.9), claim("c", "a.py", None, 0.4)]
    assert len(dedupe.one_per_site(claims)) == 2


def test_ties_are_broken_by_line_so_the_choice_is_stable():
    first = dedupe.one_per_site([claim("c", "a.py", 20, 0.8), claim("c", "a.py", 18, 0.8)])
    again = dedupe.one_per_site([claim("c", "a.py", 18, 0.8), claim("c", "a.py", 20, 0.8)])
    assert [c["line"] for c in first] == [c["line"] for c in again] == [18]
