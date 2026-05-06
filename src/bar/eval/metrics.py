"""Dependency-free prediction metrics."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    pairs = list(zip(labels, scores))
    positives = [score for label, score in pairs if label == 1]
    negatives = [score for label, score in pairs if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def average_precision(labels: Iterable[int], scores: Iterable[float]) -> float:
    ranked = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    positives = sum(1 for label, _ in ranked if label == 1)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def precision_recall_curve(labels: Iterable[int], scores: Iterable[float]) -> List[Tuple[float, float, float]]:
    ranked = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    positives = sum(1 for label, _ in ranked if label == 1)
    if positives == 0:
        return []
    tp = 0
    fp = 0
    curve = []
    for label, score in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        curve.append((precision, recall, score))
    return curve
