"""Citation precision utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from bar.kg.graph_store import InMemoryGraphStore
from bar.reasoning.prompts import CITATION_JUDGE_PROMPT
from bar.schemas import CohortSample


class CitationJudge(Protocol):
    def judge(self, prompt: str) -> str:
        ...


@dataclass
class CitationDecision:
    edge_id: str
    relevant: bool
    reason: str = ""


def citation_precision(decisions: Iterable[CitationDecision]) -> float:
    decisions = list(decisions)
    if not decisions:
        return 0.0
    return sum(1 for item in decisions if item.relevant) / len(decisions)


def build_citation_judge_prompt(sample: CohortSample, edge_id: str, store: InMemoryGraphStore) -> str:
    edge = store.edge(edge_id)
    if edge is None:
        raise ValueError(f"unknown edge id: {edge_id}")
    return CITATION_JUDGE_PROMPT.format(
        patient_codes=sample.diagnosis_codes,
        target_disease=sample.target_disease_id,
        cited_edge=f"{edge.head}; {edge.relation}; {edge.tail}",
        support_score=edge.support_score,
    )


def judge_citations(
    sample: CohortSample,
    edge_ids: List[str],
    store: InMemoryGraphStore,
    judge: CitationJudge,
) -> List[CitationDecision]:
    decisions = []
    for edge_id in edge_ids:
        prompt = build_citation_judge_prompt(sample, edge_id, store)
        response = judge.judge(prompt).strip()
        relevant = response.upper().startswith("RELEVANT")
        decisions.append(CitationDecision(edge_id=edge_id, relevant=relevant, reason=response))
    return decisions
