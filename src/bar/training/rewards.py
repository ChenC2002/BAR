"""Paired gain, acquisition cost, and citation-integrity reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from bar.kg.graph_store import InMemoryGraphStore
from bar.schemas import ReasoningBudget, ReasoningTrace


@dataclass
class RewardWeights:
    gain: float = 1.0
    cost: float = 0.1
    citation_integrity: float = 0.25


def binary_cross_entropy(probability: float, label: int, eps: float = 1e-8) -> float:
    p = min(max(probability, eps), 1.0 - eps)
    return -(label * math.log(p) + (1 - label) * math.log(1.0 - p))


def paired_gain(raw_loss: float, augmented_loss: float) -> float:
    return raw_loss - augmented_loss


def normalized_acquisition_cost(trace: ReasoningTrace, max_budget: float) -> float:
    return trace.total_cost / max(max_budget, 1e-8)


def citation_integrity(
    edge_ids: Iterable[str],
    store: InMemoryGraphStore,
    min_support: float = 0.55,
    citation_cap: int = 5,
) -> Dict[str, float]:
    edge_id_list = list(edge_ids)
    if not edge_id_list:
        return {
            "citation_integrity": 0.0,
            "exists": 0.0,
            "support_compliance": 0.0,
            "non_redundancy": 0.0,
            "has_provenance": 0.0,
        }

    known_edges = [store.edge(edge_id) for edge_id in edge_id_list if store.edge(edge_id) is not None]
    exists = len(known_edges) / len(edge_id_list)
    support_hits = [edge for edge in known_edges if edge.support_score >= min_support]
    support_compliance = len(support_hits) / max(len(known_edges), 1)
    groups: List[str] = [edge.equiv_group_id or edge.edge_id for edge in known_edges]
    non_redundancy = len(set(groups)) / max(len(groups), 1)
    has_provenance = sum(1 for edge in known_edges if edge.provenance) / max(len(known_edges), 1)
    cap_ok = float(len(edge_id_list) <= citation_cap)
    integrity = min(exists, support_compliance, non_redundancy, has_provenance, cap_ok)
    return {
        "citation_integrity": integrity,
        "exists": exists,
        "support_compliance": support_compliance,
        "non_redundancy": non_redundancy,
        "has_provenance": has_provenance,
    }


def total_reward(
    raw_loss: float,
    augmented_loss: float,
    trace: ReasoningTrace,
    store: InMemoryGraphStore,
    max_budget: float,
    weights: RewardWeights | None = None,
    min_support: float = 0.55,
    citation_cap: int = 5,
) -> Dict[str, float]:
    weights = weights or RewardWeights()
    quality = citation_integrity(
        trace.citation_edge_ids,
        store,
        min_support=min_support,
        citation_cap=citation_cap,
    )
    gain_value = paired_gain(raw_loss, augmented_loss)
    cost_value = normalized_acquisition_cost(trace, max_budget=max_budget)
    reward = (
        weights.gain * gain_value
        - weights.cost * cost_value
        + weights.citation_integrity * quality["citation_integrity"]
    )
    return {
        "reward": reward,
        "paired_gain": gain_value,
        "normalized_acquisition_cost": cost_value,
        **quality,
    }


def budget_usage(trace: ReasoningTrace, budget: ReasoningBudget) -> Dict[str, float]:
    return {
        "total_cost": trace.total_cost,
        "budget": trace.patient_budget,
        "budget_utilization": trace.total_cost / max(trace.patient_budget, 1e-8),
        "evidence_ratio": len(trace.evidence_edge_ids) / max(budget.evidence_cap, 1),
        "citation_ratio": len(trace.citation_edge_ids) / max(budget.citation_cap, 1),
    }
