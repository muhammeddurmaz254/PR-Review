"""Tests for the scoring harness.

The corpus verifier checks the corpus. These check the thing that scores it: a
matching bug or an off-by-one in the cascade would silently move every number
the project reports, and nothing downstream would notice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baselines
import candidates as candidate_module
import metrics
from adapters import load_cases
from matching import admissible, match_all, match_one
from schema import CASCADE, PRIMARY, Case, MatchConfig, Prediction, Span


@pytest.fixture(scope="session")
def cases() -> list[Case]:
    return load_cases()


@pytest.fixture(scope="session")
def oracle(cases) -> list[Prediction]:
    """A run that reports every required in-scope label at its exact anchor."""
    return [
        Prediction(case.case_id, label.span, label.type, 1.0, "oracle", "oracle")
        for case in cases for label in case.scored_labels
    ]


def test_span_distance_is_symmetric_and_zero_on_overlap():
    left, right = Span("a.py", 10, 20), Span("a.py", 15, 25)
    assert left.distance(right) == right.distance(left) == 0
    far = Span("a.py", 24, 24)
    assert Span("a.py", 20, 20).distance(far) == 4
    assert Span("b.py", 20, 20).distance(far) is None


def test_tolerance_boundary_is_inclusive():
    label_span = Span("a.py", 10, 10)
    for offset, expected in ((3, True), (4, False)):
        prediction = Prediction("c", Span("a.py", 10 + offset, 10 + offset), "authz")
        label = _label(label_span)
        assert (admissible(prediction, label, PRIMARY) is not None) is expected


def _label(span: Span, defect_type: str = "authz", required: bool = True, in_scope: bool = True):
    from schema import Label
    return Label(
        finding_id="f", case_id="c", type=defect_type, family="authorization",
        in_scope=in_scope, required=required, role="primary", spans=(span,),
    )


def test_matching_is_one_to_one():
    labels = [_label(Span("a.py", 10, 10)), _label(Span("a.py", 11, 11))]
    predictions = [Prediction("c", Span("a.py", 10, 10), "authz")]
    result = match_one("c", predictions, labels, PRIMARY)
    assert len(result.matches) == 1
    assert len(result.unmatched_labels) == 1

    predictions = [Prediction("c", Span("a.py", 10, 10), "authz"), Prediction("c", Span("a.py", 10, 10), "authz")]
    result = match_one("c", predictions, [labels[0]], PRIMARY)
    assert len(result.matches) == 1
    assert len(result.unmatched_predictions) == 1


def test_matching_ignores_prediction_order(cases, oracle):
    forward = metrics.score(cases, match_all(cases, oracle, PRIMARY))
    backward = metrics.score(cases, match_all(cases, list(reversed(oracle)), PRIMARY))
    assert forward.as_dict() == backward.as_dict()


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


@pytest.mark.parametrize("config", CASCADE, ids=[rung.label() for rung in CASCADE])
def test_greedy_matching_is_optimal_on_the_corpus(cases, config: MatchConfig):
    """Greedy can be suboptimal in theory; on this corpus it is not.

    The claim in ``matching`` that greedy and optimal coincide has to be checked
    against something, so every cascade rung is compared with a maximum
    bipartite matching over the same admissible pairs.
    """
    for case in cases:
        predictions = [
            Prediction(case.case_id, label.span, label.type, 1.0, "oracle")
            for label in case.labels
        ] + [Prediction(case.case_id, Span(name, 1, 1), "authz", 0.5, "noise")
             for name in sorted(case.head_files)[:2]]
        result = match_one(case.case_id, predictions, case.labels, config)
        ordered = sorted(predictions, key=lambda p: (-p.confidence, p.span, p.type, p.detector))
        pairs = [
            (prediction_index, label_index)
            for prediction_index, prediction in enumerate(ordered)
            for label_index, label in enumerate(case.labels)
            if admissible(prediction, label, config) is not None
        ]
        assert len(result.matches) == _max_bipartite(pairs, len(ordered)), case.case_id


def test_oracle_scores_perfectly(cases, oracle):
    result = metrics.evaluate(cases, oracle, PRIMARY)
    assert result["primary"]["recall"] == 1.0
    assert result["primary"]["precision"] == 1.0
    assert result["primary"]["fn"] == 0
    assert result["pairwise_in_scope"]["accuracy"] == 1.0
    assert result["pr_level_in_scope"]["tn"] == 46


def test_cascade_recall_is_monotone(cases, oracle):
    """Tightening the rung can never find labels a looser rung missed."""
    drifted = []
    for case in cases:
        if not case.labels:
            continue
        label = case.labels[0]
        line = label.span.start_line + 7
        drifted.append(Prediction(case.case_id, Span(label.span.file, line, line), label.type, 0.4, "drift"))
    recalls = [card.recall for card in metrics.cascade(cases, oracle + drifted).values()]
    assert recalls == sorted(recalls, reverse=True)


def test_out_of_scope_matches_are_neutral(cases):
    """Reporting a real defect the current scope excludes must not cost precision."""
    case = next(case for case in cases if any(not label.in_scope for label in case.labels))
    label = next(label for label in case.labels if not label.in_scope)
    card = metrics.score(cases, match_all(cases, [Prediction(case.case_id, label.span, label.type)], PRIMARY))
    assert card.neutral_out_scope == 1
    assert card.false_alarm == 0
    assert card.true_positive == 0


def test_every_scored_label_sits_on_a_touched_line(cases):
    """Bitbucket can only annotate changed lines, so an unreachable label is a corpus bug."""
    for case in cases:
        for label in case.scored_labels:
            assert case.touches(label.span), f"{label.finding_id} is outside the diff"


def test_no_baseline_beats_the_corpus(cases):
    """The honest baselines must stay weak; if one gets strong the corpus is gameable."""
    for name, build in baselines.BASELINES.items():
        if name in baselines.LEAKY:
            continue
        card = metrics.score(cases, match_all(cases, build(cases), PRIMARY))
        assert card.recall < 0.4, f"{name} reaches {card.recall:.0%} without understanding anything"


def test_every_baseline_stays_silent_on_no_clean_twin(cases):
    """Pairwise accuracy is the defence against always-defective strategies."""
    for name, build in baselines.BASELINES.items():
        results = match_all(cases, build(cases), PRIMARY)
        assert metrics.pairwise_accuracy(cases, results)["accuracy"] == 0.0, name


def test_enumeration_is_deterministic(cases):
    case = next(case for case in cases if case.case_id == "authz-001-buggy")
    first = candidate_module.enumerate_case(case)
    second = candidate_module.enumerate_case(case)
    assert [(c.type, c.detector, c.focus, c.region) for c in first] == \
           [(c.type, c.detector, c.focus, c.region) for c in second]


def test_candidates_stay_inside_the_diff(cases):
    """An enumerated site outside the diff cannot be annotated and must not be offered."""
    for case in cases[:20]:
        for candidate in candidate_module.enumerate_case(case):
            assert case.touches(candidate.focus) or case.touches(candidate.region), \
                f"{case.case_id}: {candidate.detector} at {candidate.focus}"
