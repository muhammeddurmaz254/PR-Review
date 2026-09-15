"""One finding per code site, and the rules only where the model said nothing."""
from prdetect.detect import dedupe


def claim(case_id, file, line, confidence, type_="x"):
    return {"case_id": case_id, "file": file, "line": line, "confidence": confidence, "type": type_}


def test_two_claims_on_one_site_become_the_more_confident_one():
    claims = [claim("c", "a.py", 53, 0.7, "path_traversal"), claim("c", "a.py", 57, 0.9, "command_injection")]
    assert [c["type"] for c in dedupe.one_per_site(claims)] == ["command_injection"]


def test_a_second_finding_far_enough_away_is_kept():
    assert len(dedupe.one_per_site([claim("c", "a.py", 10, 0.9), claim("c", "a.py", 40, 0.8)])) == 2


def test_the_same_line_in_another_file_is_another_site():
    assert len(dedupe.one_per_site([claim("c", "a.py", 10, 0.9), claim("c", "b.py", 10, 0.8)])) == 2


def test_pull_requests_do_not_shadow_each_other():
    assert len(dedupe.one_per_site([claim("one", "a.py", 10, 0.9), claim("two", "a.py", 11, 0.8)])) == 2


def test_the_order_given_is_kept():
    claims = [claim("c", "a.py", 10, 0.5), claim("c", "b.py", 10, 0.9), claim("c", "b.py", 12, 0.4)]
    assert [(c["file"], c["line"]) for c in dedupe.one_per_site(claims)] == [("a.py", 10), ("b.py", 10)]


def test_a_claim_with_no_line_is_never_merged_away():
    assert len(dedupe.one_per_site([claim("c", "a.py", None, 0.9), claim("c", "a.py", None, 0.4)])) == 2


def test_ties_are_broken_by_line_so_the_choice_is_stable():
    first = dedupe.one_per_site([claim("c", "a.py", 20, 0.8), claim("c", "a.py", 18, 0.8)])
    again = dedupe.one_per_site([claim("c", "a.py", 18, 0.8), claim("c", "a.py", 20, 0.8)])
    assert [c["line"] for c in first] == [c["line"] for c in again] == [18]


def test_a_rule_speaks_only_where_the_model_did_not():
    model = [claim("c", "a.py", 52, 0.95, "weak_crypto_primitive")]
    extra = [claim("c", "a.py", 52, 0.9, "hardcoded_credential"), claim("c", "b.py", 10, 0.9, "unsafe_default")]
    kept = dedupe.fill_gaps(model, extra)
    assert [(c["file"], c["type"]) for c in kept] == [("a.py", "weak_crypto_primitive"), ("b.py", "unsafe_default")]


def test_fill_gaps_keeps_the_published_order():
    model = [claim("c", "a.py", 10, 0.9), claim("c", "b.py", 10, 0.8)]
    kept = dedupe.fill_gaps(model, [claim("c", "c.py", 1, 0.9)])
    assert [c["file"] for c in kept] == ["a.py", "b.py", "c.py"]
