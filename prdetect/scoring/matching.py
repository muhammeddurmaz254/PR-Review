"""One-to-one assignment between findings and labels.

A finding satisfies at most one label and a label takes at most one finding.
Matching is greedy on line distance: the closest admissible pair is taken first,
ties broken by confidence and then by position, so the result never depends on
the order findings arrive in. Admissible pairs are confined to one file and a few
lines apart, where greedy and optimal assignment coincide; the tests check that
on both repositories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from prdetect.cases import Case, Label, Prediction

TypeMode = Literal["exact", "family", "none"]


@dataclass(frozen=True)
class MatchConfig:
    """How far a finding may sit from a label, and whether its kind has to agree.

    `type_mode` is `none` (location only), `family` or `exact`. `tolerance` is in
    lines; None drops the line constraint.
    """

    tolerance: int | None = 10
    type_mode: TypeMode = "exact"

    def label_key(self, label: Label) -> str | None:
        return {"exact": label.type, "family": label.family, "none": None}[self.type_mode]

    def prediction_key(self, prediction: Prediction) -> str | None:
        return {"exact": prediction.type, "family": prediction.family, "none": None}[self.type_mode]


@dataclass(frozen=True)
class Match:
    prediction: Prediction
    label: Label
    distance: int


@dataclass
class MatchResult:
    """What one pull request's findings were matched to."""

    case_id: str
    matches: tuple[Match, ...]
    unmatched_predictions: tuple[Prediction, ...]
    unmatched_labels: tuple[Label, ...]


def admissible(prediction: Prediction, label: Label, config: MatchConfig) -> int | None:
    """Line distance when the pair may be matched, else None.

    A label with equivalent locations is satisfied by its nearest one.
    """
    if config.prediction_key(prediction) != config.label_key(label):
        return None
    distances = [distance for span in label.spans
                 if (distance := prediction.span.distance(span)) is not None]
    if not distances:
        return None
    best = min(distances)
    if config.tolerance is not None and best > config.tolerance:
        return None
    return best


def _order(predictions: Sequence[Prediction]) -> list[Prediction]:
    return sorted(predictions, key=lambda p: (-p.confidence, p.span, p.type, p.detector))


def match_one(case_id: str, predictions: Sequence[Prediction], labels: Sequence[Label],
              config: MatchConfig) -> MatchResult:
    ordered = _order(predictions)
    pairs = []
    for prediction_index, prediction in enumerate(ordered):
        for label_index, label in enumerate(labels):
            distance = admissible(prediction, label, config)
            if distance is not None:
                pairs.append((distance, prediction_index, label_index))
    pairs.sort()
    used_predictions: set[int] = set()
    used_labels: set[int] = set()
    matches = []
    for distance, prediction_index, label_index in pairs:
        if prediction_index in used_predictions or label_index in used_labels:
            continue
        used_predictions.add(prediction_index)
        used_labels.add(label_index)
        matches.append(Match(ordered[prediction_index], labels[label_index], distance))
    return MatchResult(
        case_id=case_id,
        matches=tuple(matches),
        unmatched_predictions=tuple(p for i, p in enumerate(ordered) if i not in used_predictions),
        unmatched_labels=tuple(l for i, l in enumerate(labels) if i not in used_labels),
    )


def match_all(cases: Iterable[Case], predictions: Iterable[Prediction],
              config: MatchConfig) -> dict[str, MatchResult]:
    """Match every case, including those with no findings and no labels."""
    by_case: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        by_case.setdefault(prediction.case_id, []).append(prediction)
    results = {}
    for case in cases:
        results[case.case_id] = match_one(case.case_id, by_case.get(case.case_id, []), case.labels, config)
    unknown = sorted(set(by_case) - set(results))
    if unknown:
        raise SystemExit(f"predictions reference unknown cases: {', '.join(unknown[:5])}")
    return results
