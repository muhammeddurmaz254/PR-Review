"""Scope-aware scoring, the localization cascade, and the diagnostic breakdowns.

Every prediction falls into exactly one bucket:

``true_positive``      matched a required in-scope label
``neutral_optional``   matched an in-scope label the corpus marks not required
``neutral_out_scope``  matched a label outside the current detector scope
``false_alarm``        matched nothing

Only the first and last enter precision and recall; the two neutral buckets are
reported alongside so that detecting a real but presently unsupported defect
neither helps nor hurts. The recall denominator is the required in-scope labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from matching import MatchResult, match_all
from schema import CASCADE, IN_SCOPE_TYPES, Case, MatchConfig, Prediction

DEFAULT_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class ScoreCard:
    true_positive: int = 0
    false_alarm: int = 0
    false_negative: int = 0
    neutral_optional: int = 0
    neutral_out_scope: int = 0

    @property
    def predictions(self) -> int:
        return self.true_positive + self.false_alarm + self.neutral_optional + self.neutral_out_scope

    @property
    def precision(self) -> float:
        return _ratio(self.true_positive, self.true_positive + self.false_alarm)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positive, self.true_positive + self.false_negative)

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.true_positive, "fp": self.false_alarm, "fn": self.false_negative,
            "neutral_optional": self.neutral_optional, "neutral_out_of_scope": self.neutral_out_scope,
            "predictions": self.predictions, "precision": round(self.precision, 4),
            "recall": round(self.recall, 4), "f1": round(self.f1, 4),
        }


def score(cases: Sequence[Case], results: dict[str, MatchResult]) -> ScoreCard:
    card = ScoreCard()
    for case in cases:
        result = results[case.case_id]
        for match in result.matches:
            if match.label.scored:
                card.true_positive += 1
            elif match.label.in_scope:
                card.neutral_optional += 1
            else:
                card.neutral_out_scope += 1
        card.false_alarm += len(result.unmatched_predictions)
        card.false_negative += sum(1 for label in result.unmatched_labels if label.scored)
    return card


def cascade(
    cases: Sequence[Case], predictions: Sequence[Prediction], threshold: float = 0.0,
    rungs: Sequence[MatchConfig] = CASCADE,
) -> dict[str, ScoreCard]:
    """Score every rung. First rung minus last rung is the localization loss."""
    return {rung.label(): score(cases, match_all(cases, predictions, rung, threshold)) for rung in rungs}


@dataclass
class DetectionCard:
    """Case level: did the run flag this pull request at all?"""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    def as_dict(self) -> dict:
        return {
            "tp": self.true_positive, "fp": self.false_positive,
            "tn": self.true_negative, "fn": self.false_negative,
            "precision": round(_ratio(self.true_positive, self.true_positive + self.false_positive), 4),
            "recall": round(_ratio(self.true_positive, self.true_positive + self.false_negative), 4),
            "accuracy": round(_ratio(self.true_positive + self.true_negative, self.total), 4),
        }


def pr_level(cases: Sequence[Case], results: dict[str, MatchResult], scope: str = "in_scope") -> DetectionCard:
    """Flagging accuracy.

    ``in_scope`` restricts positives to cases carrying a required in-scope label
    and only counts reports of in-scope types; that is the primary metric.
    Defective cases whose only labels are out of scope are excluded from it
    rather than counted as negatives. ``all`` uses every defective case and
    every report.
    """
    card = DetectionCard()
    for case in cases:
        result = results[case.case_id]
        if scope == "in_scope":
            positive = bool(case.scored_labels)
            if not positive and case.is_defective:
                continue
            reported = (*(match.prediction for match in result.matches), *result.unmatched_predictions)
            flagged = any(prediction.type in IN_SCOPE_TYPES for prediction in reported)
        else:
            positive = case.is_defective
            flagged = bool(result.matches) or bool(result.unmatched_predictions)
        if positive and flagged:
            card.true_positive += 1
        elif positive:
            card.false_negative += 1
        elif flagged:
            card.false_positive += 1
        else:
            card.true_negative += 1
    return card


def pairwise_accuracy(cases: Sequence[Case], results: dict[str, MatchResult], scope: str = "in_scope") -> dict:
    """Share of features where the defective variant is caught and the clean twin stays quiet.

    An always-defective strategy cannot score here: it fails every clean twin.
    """
    by_pair: dict[str, dict[str, Case]] = {}
    for case in cases:
        if case.pair_id:
            by_pair.setdefault(case.pair_id, {})[case.variant] = case
    correct = considered = 0
    missed: list[str] = []
    noisy: list[str] = []
    for pair_id, variants in sorted(by_pair.items()):
        buggy, clean = variants.get("buggy"), variants.get("clean")
        if buggy is None or clean is None:
            continue
        if scope == "in_scope" and not buggy.scored_labels:
            continue
        considered += 1
        buggy_result, clean_result = results[buggy.case_id], results[clean.case_id]
        if scope == "in_scope":
            detected = any(match.label.scored for match in buggy_result.matches)
        else:
            detected = bool(buggy_result.matches)
        quiet = not clean_result.unmatched_predictions
        if detected and quiet:
            correct += 1
        elif not detected:
            missed.append(pair_id)
        else:
            noisy.append(pair_id)
    return {
        "pairs": considered, "correct": correct, "accuracy": round(_ratio(correct, considered), 4),
        "missed": missed, "noisy_clean_twin": noisy,
    }


def false_alarms(cases: Sequence[Case], results: dict[str, MatchResult]) -> dict:
    """Noise budget: unmatched reports per pull request, split by case kind."""
    groups: dict[str, list[int]] = {"all": [], "defective": [], "clean_twin": [], "standalone_trap": []}
    on_distractor = 0
    for case in cases:
        unmatched = results[case.case_id].unmatched_predictions
        groups["all"].append(len(unmatched))
        if case.is_defective:
            groups["defective"].append(len(unmatched))
        elif case.pair_id:
            groups["clean_twin"].append(len(unmatched))
        else:
            groups["standalone_trap"].append(len(unmatched))
        for prediction in unmatched:
            for distractor in case.distractors:
                distance = prediction.span.distance(distractor.span)
                if distance is not None and distance <= 3:
                    on_distractor += 1
                    break
    summary: dict = {
        name: {"cases": len(values), "total": sum(values), "per_pr": round(mean(values), 4) if values else 0.0}
        for name, values in groups.items()
    }
    summary["on_distractor"] = on_distractor
    return summary


def breakdown(cases: Sequence[Case], results: dict[str, MatchResult], key: str) -> dict[str, dict]:
    """Recall over required in-scope labels, grouped by type, family or difficulty."""
    buckets: dict[str, ScoreCard] = {}
    for case in cases:
        matched = set(results[case.case_id].matched_labels)
        for label in case.labels:
            if not label.scored:
                continue
            name = {"type": label.type, "family": label.family, "difficulty": case.difficulty}[key]
            card = buckets.setdefault(name, ScoreCard())
            if label in matched:
                card.true_positive += 1
            else:
                card.false_negative += 1
    return {
        name: {"tp": card.true_positive, "fn": card.false_negative, "recall": round(card.recall, 4)}
        for name, card in sorted(buckets.items())
    }


def threshold_sweep(
    cases: Sequence[Case], predictions: Sequence[Prediction], config: MatchConfig,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        results = match_all(cases, predictions, config, threshold)
        card = score(cases, results)
        rows.append({
            "threshold": threshold, **card.as_dict(),
            "fp_per_pr": round(_ratio(card.false_alarm, len(cases)), 4),
            "pairwise": pairwise_accuracy(cases, results)["accuracy"],
        })
    return rows


@dataclass
class Stage:
    """One pipeline stage and the predictions that survived it."""

    name: str
    predictions: tuple[Prediction, ...]


def stage_losses(cases: Sequence[Case], stages: Sequence[Stage], config: MatchConfig) -> list[dict]:
    """Per-stage kept and lost true positives, and added false alarms.

    An F1 of 41 percent does not say where to spend a week; this table does.
    """
    rows: list[dict] = []
    previous_caught: set[str] = set()
    previous_count = 0
    previous_fp = 0
    for index, stage in enumerate(stages):
        results = match_all(cases, stage.predictions, config)
        card = score(cases, results)
        caught = {
            match.label.finding_id
            for result in results.values() for match in result.matches if match.label.scored
        }
        rows.append({
            "stage": stage.name,
            "entered": previous_count if index else len(stage.predictions),
            "left": len(stage.predictions),
            "tp_kept": len(caught),
            "tp_lost": len(previous_caught - caught) if index else 0,
            "fp": card.false_alarm,
            "fp_added": card.false_alarm - previous_fp if index else card.false_alarm,
        })
        previous_caught, previous_count, previous_fp = caught, len(stage.predictions), card.false_alarm
    return rows


def evaluate(
    cases: Sequence[Case], predictions: Sequence[Prediction], config: MatchConfig, threshold: float = 0.0,
) -> dict:
    """The full metric set for one prediction source."""
    results = match_all(cases, predictions, config, threshold)
    return {
        "primary": {
            "tolerance": config.tolerance, "type_mode": config.type_mode, "threshold": threshold,
            **score(cases, results).as_dict(),
        },
        "cascade": {name: card.as_dict() for name, card in cascade(cases, predictions, threshold).items()},
        "pr_level_in_scope": pr_level(cases, results, "in_scope").as_dict(),
        "pr_level_all": pr_level(cases, results, "all").as_dict(),
        "pairwise_in_scope": pairwise_accuracy(cases, results, "in_scope"),
        "pairwise_all": pairwise_accuracy(cases, results, "all"),
        "false_alarms": false_alarms(cases, results),
        "by_type": breakdown(cases, results, "type"),
        "by_family": breakdown(cases, results, "family"),
        "by_difficulty": breakdown(cases, results, "difficulty"),
        "threshold_sweep": threshold_sweep(cases, predictions, config),
    }
