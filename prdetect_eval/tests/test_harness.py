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
import metrics
import steps
from adapters import DEFAULT_EVAL, load_cases
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
    loose = MatchConfig(tolerance=3, type_mode="exact")
    for offset, expected in ((3, True), (4, False)):
        prediction = Prediction("c", Span("a.py", 10 + offset, 10 + offset), "authz")
        assert (admissible(prediction, _label(label_span), loose) is not None) is expected


def test_primary_requires_landing_inside_the_region():
    """The primary rung asks whether the report points at the code, not near it."""
    region = _label(Span("a.py", 10, 20))
    inside = Prediction("c", Span("a.py", 15, 15), "authz")
    outside = Prediction("c", Span("a.py", 22, 22), "authz")
    assert admissible(inside, region, PRIMARY) == 0
    assert admissible(outside, region, PRIMARY) is None


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
    # The in-scope universe skips defective cases whose labels are all out of
    # scope, so its negatives are exactly the clean cases. Derived rather than
    # hardcoded: a growing corpus must not need the constant edited.
    assert result["pr_level_in_scope"]["tn"] == sum(1 for case in cases if not case.is_defective)


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


def test_every_scored_label_is_reachable(cases):
    """A label must sit on changed code, unless the change was a deletion.

    A control removed by the pull request leaves no line behind to annotate, and
    that shape is a third of what makes real review hard, so it is labelled
    rather than excluded. Everything else has to be inside the diff.
    """
    for case in cases:
        for label in case.scored_labels:
            assert case.touches(label.span) or label.pure_deletion, \
                f"{label.finding_id} is outside the diff and not marked pure_deletion"


def test_labels_carry_a_focus_and_a_title(cases):
    """The step file reports both, and a missing one silently reads as a zero."""
    for case in cases:
        for label in case.scored_labels:
            assert label.focus is not None, label.finding_id
            assert label.focus.file == label.span.file, label.finding_id
            assert label.title.strip(), label.finding_id


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





def test_balanced_accuracy_is_half_for_any_input_blind_strategy():
    """The point of the PR-level metric: flagging everything, flagging nothing
    and flipping a coin all score 0.5, on any class balance."""
    always = metrics.DetectionCard(true_positive=25, false_positive=25, true_negative=0, false_negative=0)
    never = metrics.DetectionCard(true_positive=0, false_positive=0, true_negative=25, false_negative=25)
    lopsided = metrics.DetectionCard(true_positive=90, false_positive=10, true_negative=0, false_negative=0)
    for card in (always, never, lopsided):
        assert card.as_dict()["balanced_accuracy"] == 0.5
    # ...while plain accuracy hands the lopsided corpus a free 90%.
    assert lopsided.as_dict()["accuracy"] == 0.9


def test_specificity_counts_the_clean_pull_requests_left_alone():
    card = metrics.DetectionCard(true_positive=13, false_positive=10, true_negative=15, false_negative=12)
    assert card.as_dict()["specificity"] == 0.6
    assert card.as_dict()["balanced_accuracy"] == 0.56


def test_the_step_file_leads_with_the_pull_request_verdict():
    """It is the question the product answers; burying it under the finding
    table is how a run reads as better than it is."""
    cases = load_cases(DEFAULT_EVAL)
    predictions = baselines.flag_everything(cases)
    result = metrics.evaluate(cases, predictions, MatchConfig(tolerance=3, type_mode="exact"))
    text = steps.render("t", {"cases": len(cases)}, result)
    assert text.index("PR seviyesi") < text.index("Bulgu seviyesi")
    assert "0.50 = girdiyi yok saymak" in text
