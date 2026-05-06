"""QUERY_GUIDE and QUERY_EXPAND over a refined evidence graph."""

from __future__ import annotations

from typing import List, Optional

from bar.kg.graph_store import InMemoryGraphStore
from bar.schemas import GraphQuery, QueryResult, ReasoningBudget


class EvidenceGraphQueryEngine:
    """Executable graph API used by the plan-guided navigation step."""

    def __init__(self, store: InMemoryGraphStore, budget: ReasoningBudget) -> None:
        self.store = store
        self.budget = budget

    def execute(self, query: GraphQuery) -> QueryResult:
        if query.query_type == "QUERY_GUIDE":
            return self.query_guide(query.anchor_id, top_k=query.top_k)
        if query.query_type == "QUERY_EXPAND":
            return self.query_expand(query.concept_id, relation=query.relation, top_k=query.top_k)
        return QueryResult(valid=False, terminal=True, message=f"unknown query_type {query.query_type}")

    def query_guide(self, anchor_id: Optional[str], top_k: int) -> QueryResult:
        if not anchor_id:
            return QueryResult(valid=False, message="QUERY_GUIDE requires anchor_id")
        edges = self.store.frontier(anchor_id, top_k=top_k)
        return QueryResult(
            valid=True,
            candidate_edge_ids=[edge.edge_id for edge in edges],
            cost=self.budget.query_cost,
            payload={"anchor_id": anchor_id, "source": "expansion_guide_or_incident_lookup"},
        )

    def query_expand(self, concept_id: Optional[str], relation: Optional[str], top_k: int) -> QueryResult:
        if not concept_id:
            return QueryResult(valid=False, message="QUERY_EXPAND requires concept_id")
        edges = self.store.neighbors(concept_id, relation=relation, top_k=top_k)
        return QueryResult(
            valid=True,
            candidate_edge_ids=[edge.edge_id for edge in edges],
            cost=self.budget.query_cost + self.budget.expand_cost,
            payload={"concept_id": concept_id, "relation": relation},
        )

    def candidate_payload(self, edge_ids: List[str]) -> List[dict]:
        payload = []
        for edge_id in edge_ids:
            edge = self.store.edge(edge_id)
            if edge is None:
                continue
            payload.append(
                {
                    "edge_id": edge.edge_id,
                    "head": edge.head,
                    "relation": edge.relation,
                    "tail": edge.tail,
                    "support_score": edge.support_score,
                    "equiv_group_id": edge.equiv_group_id,
                    "provenance_count": len(edge.provenance),
                }
            )
        return payload
