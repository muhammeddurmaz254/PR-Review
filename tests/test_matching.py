"""One-to-one matching and the score built on it.

A matching bug would silently move every number a report prints, so the matcher
is checked on its own and against a maximum bipartite matching on both repositories.
"""
from __future__ import annotations

import pytest

from prdetect.cases import Case, Label, Prediction, Span
from prdetect.scoring import metrics
from prdetect.scoring.matching import MatchConfig, admissible, match_all, match_one
from tests.repositories import NARROW, REPOSITORIES, cases

RUNGS = (MatchConfig(None, "none"), MatchConfig(None, "family"), MatchConfig(None, "exact"),
         MatchConfig(10, "exact"), MatchConfig(3, "exact"), MatchConfig(0, "exact"))


def _label(span: Span, kind: str = "missing_authz_check", required: bool = True, in_scope: bool = True,
           case_id: str = "c") -> Label:
    return Label(finding_id=f"{case_id}-f1", case_id=case_id, type=kind, family="authz",
                 in_scope=in_scope, required=required, spans=(span,))


def _oracle(loaded: list[Case]) -> list[Prediction]:
    return [Prediction(case.case_id, label.span, label.type, 1.0, "oracle")
            for case in loaded for label in case.scored_labels]


def test_span_distance_is_symmetric_and_zero_on_overlap():
    left, right = Span("a.py", 10, 20), Span("a.py", 15, 25)
    assert left.distance(right) == right.distance(left) == 0
    far = Span("a.py", 24, 24)
    assert Span("a.py", 20, 20).distance(far) == 4
    assert Span("b.py", 20, 20).distance(far) is None


def test_tolerance_boundary_is_inclusive():
    loose = MatchConfig(tolerance=3, type_mode="exact")
    for offset, expected in ((3, True), (4, False)):
        prediction = Prediction("c", Span("a.py", 10 + offset, 10 + offset), "missing_authz_check")
        assert (admissible(prediction, _label(Span("a.py", 10, 10)), loose) is not None) is expected


def test_tolerance_zero_requires_landing_inside_the_label():
    region = _label(Span("a.py", 10, 20))
    exact = MatchConfig(tolerance=0, type_mode="exact")
    assert admissible(Prediction("c", Span("a.py", 15, 15), "missing_authz_check"), region, exact) == 0
    assert admissible(Prediction("c", Span("a.py", 22, 22), "missing_authz_check"), region, exact) is None


def test_the_type_modes():
    label = _label(Span("a.py", 10, 10))
    same_family = Prediction("c", Span("a.py", 10, 10), "mass_assignment")
    assert admissible(same_family, label, MatchConfig(0, "exact")) is None
    assert admissible(same_family, label, MatchConfig(0, "family")) == 0
    assert admissible(Prediction("c", Span("a.py", 10, 10), "xss"), label, MatchConfig(0, "none")) == 0


def test_matching_is_one_to_one():
    labels = [_label(Span("a.py", 10, 10)), _label(Span("a.py", 11, 11))]
    exact = MatchConfig(tolerance=0, type_mode="exact")
    result = match_one("c", [Prediction("c", Span("a.py", 10, 10), "missing_authz_check")], labels, exact)
    assert len(result.matches) == 1 and len(result.unmatched_labels) == 1

    twice = [Prediction("c", Span("a.py", 10, 10), "missing_authz_check")] * 2
    result = match_one("c", twice, [labels[0]], exact)
    assert len(result.matches) == 1 and len(result.unmatched_predictions) == 1


def test_matching_ignores_prediction_order():
    loaded = cases(NARROW)
    oracle = _oracle(loaded)
    config = MatchConfig(tolerance=0, type_mode="exact")
    forward = metrics.score(loaded, match_all(loaded, oracle, config))
    backward = metrics.score(loaded, match_all(loaded, list(reversed(oracle)), config))
    assert forward.as_dict() == backward.as_dict()


def test_the_oracle_scores_perfectly():
    for repo in REPOSITORIES:
        loaded = cases(repo)
        card = metrics.score(loaded, match_all(loaded, _oracle(loaded), MatchConfig(0, "exact")))
        assert (card.precision, card.recall, card.false_alarm) == (1.0, 1.0, 0), repo


def _max_bipartite(pairs: list[tuple[int, int]], left: int) -> int:
    """Kuhn's algorithm: the largest possible number of matches."""
    adjacency: dict[int, list[int]] = {index: [] for index in range(left)}
    for prediction_index, label_index in pairs:
        adjacency[prediction_index].append(label_index)
    assigned: dict[int, int] = {}

    def augment(node: int, seen: set[int]) -> bool:
        for target in adjacency[node]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target], seen):
                assigned[target] = node
                return True
        return False

    return sum(augment(node, set()) for node in range(left))


@pytest.mark.parametrize("repo", REPOSITORIES)
@pytest.mark.parametrize("config", RUNGS, ids=lambda c: f"{c.type_mode}-{c.tolerance}")
def test_greedy_matching_is_optimal(repo, config):
    for case in cases(repo):
        predictions = [Prediction(case.case_id, label.span, label.type, 1.0, "oracle") for label in case.labels]
        predictions += [Prediction(case.case_id, Span(name, 1, 1), "missing_authz_check", 0.5, "noise")
                        for name in sorted(case.head_files)[:2]]
        result = match_one(case.case_id, predictions, case.labels, config)
        ordered = sorted(predictions, key=lambda p: (-p.confidence, p.span, p.type, p.detector))
        pairs = [(i, j) for i, prediction in enumerate(ordered) for j, label in enumerate(case.labels)
                 if admissible(prediction, label, config) is not None]
        assert len(result.matches) == _max_bipartite(pairs, len(ordered)), case.case_id


def _one_case(label: Label) -> Case:
    return Case(case_id="c", pr_title="", pr_description="", changed_files=(label.span.file,),
                deleted_files=(), added_lines={label.span.file: frozenset({label.span.start_line})},
                head_files={label.span.file: "\n" * 30}, context_files={}, diff="", labels=(label,))


def test_a_label_the_key_does_not_require_is_neutral():
    label = _label(Span("app/views.py", 10, 12), required=False)
    case = _one_case(label)
    card = metrics.score([case], match_all([case], [Prediction("c", label.span, label.type)],
                                           MatchConfig(0, "exact")))
    assert (card.neutral_optional, card.true_positive, card.false_alarm, card.false_negative) == (1, 0, 0, 0)


def test_a_label_out_of_scope_is_neutral():
    label = _label(Span("README.md", 3, 3), in_scope=False)
    case = _one_case(label)
    card = metrics.score([case], match_all([case], [Prediction("c", label.span, label.type)],
                                           MatchConfig(0, "exact")))
    assert (card.neutral_out_scope, card.true_positive, card.false_alarm) == (1, 0, 0)


def test_an_equivalent_location_satisfies_the_label():
    label = Label(finding_id="c-f1", case_id="c", type="crossfile_unit_mismatch", family="data_layer",
                  in_scope=True, required=True, spans=(Span("a.py", 10, 10), Span("b.py", 40, 42)))
    case = _one_case(label)
    card = metrics.score([case], match_all([case], [Prediction("c", Span("b.py", 41, 41), label.type)],
                                           MatchConfig(0, "exact")))
    assert card.true_positive == 1


def test_a_prediction_for_an_unknown_case_is_refused():
    case = _one_case(_label(Span("a.py", 1, 1)))
    with pytest.raises(SystemExit):
        match_all([case], [Prediction("nope", Span("a.py", 1, 1), "xss")], MatchConfig())
