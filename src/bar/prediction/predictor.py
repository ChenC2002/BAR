"""Predictor and citation formation for BAR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from bar.kg.graph_store import InMemoryGraphStore
from bar.reasoning.scoring import PatientConditionedScorer
from bar.schemas import CohortSample, ReasoningTrace


@dataclass
class PredictionOutput:
    probability: float
    citation_edge_ids: List[str]
    fallback: bool
    evidence_signal: float


class HeuristicBARPredictor:
    """Dependency-free predictor with the same inputs/outputs as the paper model.

    Production training should replace this with the 2-layer GRU EHR encoder,
    edge MLP, score-weighted evidence pooling, and horizon-specific prediction
    heads from Appendix B.
    """

    def __init__(self, bias: float = -2.0, ehr_weight: float = 0.03, evidence_weight: float = 2.0) -> None:
        self.bias = bias
        self.ehr_weight = ehr_weight
        self.evidence_weight = evidence_weight
        self.scorer = PatientConditionedScorer()

    def predict(self, sample: CohortSample, trace: ReasoningTrace, store: InMemoryGraphStore) -> PredictionOutput:
        ehr_signal = self.ehr_weight * len(set(sample.diagnosis_codes + sample.history_codes))
        if trace.fallback:
            evidence_signal = 0.0
            citations: List[str] = []
        else:
            evidence_signal = self._score_weighted_pool(sample, trace.evidence_edge_ids, store)
            citations = list(trace.citation_edge_ids)
        probability = self._sigmoid(self.bias + ehr_signal + self.evidence_weight * evidence_signal)
        return PredictionOutput(
            probability=probability,
            citation_edge_ids=citations,
            fallback=trace.fallback,
            evidence_signal=evidence_signal,
        )

    def raw_predict(self, sample: CohortSample) -> float:
        ehr_signal = self.ehr_weight * len(set(sample.diagnosis_codes + sample.history_codes))
        return self._sigmoid(self.bias + ehr_signal)

    def _score_weighted_pool(self, sample: CohortSample, edge_ids: List[str], store: InMemoryGraphStore) -> float:
        if not edge_ids:
            return 0.0
        scores = [score for _, score in self.scorer.score_many(edge_ids, store, sample.anchor_node_ids)]
        return sum(scores) / max(len(scores), 1)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
