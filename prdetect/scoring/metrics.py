"""Precision, recall and F1 over matched findings.

Every finding lands in exactly one bucket:

``true_positive``      matched a scored label (in scope and required)
``neutral_optional``   matched an in-scope label the answer key does not require
``neutral_out_scope``  matched a label outside the detector's scope
``false_alarm``        matched nothing

Only the first and last enter precision and recall; finding a real defect the
answer key does not score neither helps nor hurts. Recall counts scored labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from prdetect.cases import Case
from prdetect.scoring.matching import MatchResult


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
