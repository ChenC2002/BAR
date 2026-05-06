"""Patient-conditioned evidence-quality score used inside BAR."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

from bar.kg.graph_store import InMemoryGraphStore
from bar.schemas import Edge


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class PatientConditionedScorer:
    """Dependency-free proxy for rho_t(e) = s_e * max(sigmoid(p·e_u/tau), sigmoid(p·e_v/tau)).

    The production implementation should replace ``node_relevance`` with the
    learned GRU patient state and node embeddings described in Appendix B. This
    proxy preserves the multiplicative support-and-relevance behavior so tests
    and local demos exercise the same control flow.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(temperature, 1e-6)

    def score(self, edge: Edge, patient_anchor_ids: Iterable[str]) -> float:
        anchors = set(patient_anchor_ids)
        relevance = max(
            self.node_relevance(edge.head, anchors),
            self.node_relevance(edge.tail, anchors),
        )
        return edge.support_score * relevance

    def score_many(
        self,
        edge_ids: Iterable[str],
        store: InMemoryGraphStore,
        patient_anchor_ids: Iterable[str],
    ) -> List[tuple[str, float]]:
        scored = []
        for edge_id in edge_ids:
            edge = store.edge(edge_id)
            if edge is None:
                continue
            scored.append((edge_id, self.score(edge, patient_anchor_ids)))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def node_relevance(self, node_id: str, anchors: set[str]) -> float:
        if node_id in anchors:
            return 1.0
        # Stable deterministic proxy in [0.35, 0.95].
        digest = hashlib.sha1(node_id.encode("utf-8")).hexdigest()
        raw = int(digest[:8], 16) / 0xFFFFFFFF
        centered = (raw - 0.5) / self.temperature
        return 0.35 + 0.60 * sigmoid(centered)
