"""Shared schemas for the BAR pipeline.

The project intentionally starts with dataclasses instead of a validation
dependency. If the implementation later adopts Pydantic, these classes are the
contracts to preserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class Provenance:
    source_id: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Provenance":
        return cls(
            source_id=str(data.get("source_id", "")),
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            snippet=str(data.get("snippet", "")),
            score=float(data.get("score", 0.0)),
        )

    def to_dict(self) -> JsonDict:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


@dataclass
class Node:
    node_id: str
    name: str
    node_type: str
    aliases: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Node":
        return cls(
            node_id=str(data["node_id"]),
            name=str(data.get("name", data["node_id"])),
            node_type=str(data.get("node_type", "concept")),
            aliases=list(data.get("aliases", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JsonDict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    edge_id: str
    head: str
    relation: str
    tail: str
    support_score: float = 0.0
    equiv_group_id: str = ""
    provenance: List[Provenance] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Edge":
        provenance = [Provenance.from_dict(item) for item in data.get("provenance", [])]
        return cls(
            edge_id=str(data["edge_id"]),
            head=str(data["head"]),
            relation=str(data["relation"]),
            tail=str(data["tail"]),
            support_score=float(data.get("support_score", 0.0)),
            equiv_group_id=str(data.get("equiv_group_id", data.get("edge_id", ""))),
            provenance=provenance,
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JsonDict:
        return {
            "edge_id": self.edge_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "support_score": self.support_score,
            "equiv_group_id": self.equiv_group_id or self.edge_id,
            "provenance": [item.to_dict() for item in self.provenance],
            "metadata": self.metadata,
        }


@dataclass
class EvidenceGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    equiv_groups: Dict[str, List[str]] = field(default_factory=dict)
    expansion_guide: Dict[str, List[str]] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvidenceGraph":
        nodes = {item["node_id"]: Node.from_dict(item) for item in data.get("nodes", [])}
        edges = {item["edge_id"]: Edge.from_dict(item) for item in data.get("edges", [])}
        return cls(
            nodes=nodes,
            edges=edges,
            equiv_groups={str(k): list(v) for k, v in data.get("equiv_groups", {}).items()},
            expansion_guide={str(k): list(v) for k, v in data.get("expansion_guide", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JsonDict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "equiv_groups": self.equiv_groups,
            "expansion_guide": self.expansion_guide,
            "metadata": self.metadata,
        }

    def validate(self) -> List[str]:
        errors: List[str] = []
        for edge in self.edges.values():
            if edge.head not in self.nodes:
                errors.append(f"edge {edge.edge_id} has unknown head {edge.head}")
            if edge.tail not in self.nodes:
                errors.append(f"edge {edge.edge_id} has unknown tail {edge.tail}")
            if not 0.0 <= edge.support_score <= 1.0:
                errors.append(f"edge {edge.edge_id} has support_score outside [0, 1]")
        known_edges = set(self.edges)
        for group_id, members in self.equiv_groups.items():
            missing = sorted(set(members) - known_edges)
            if missing:
                errors.append(f"equiv group {group_id} references unknown edges {missing}")
        for node_id, edge_ids in self.expansion_guide.items():
            if node_id not in self.nodes:
                errors.append(f"expansion guide references unknown node {node_id}")
            missing = sorted(set(edge_ids) - known_edges)
            if missing:
                errors.append(f"expansion guide for {node_id} references unknown edges {missing}")
        return errors


@dataclass
class PatientVisit:
    patient_id: str
    visit_id: str
    admit_time: str
    discharge_time: str
    diagnosis_codes: List[str]
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class CohortSample:
    sample_id: str
    patient_id: str
    index_visit_id: str
    index_time: str
    target_disease_id: str
    window_days: int
    label: int
    diagnosis_codes: List[str]
    history_codes: List[str]
    anchor_node_ids: List[str]
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class EvidenceItem:
    edge_id: str
    score: float
    support_score: float
    patient_relevance: float = 0.0
    reason: str = ""


@dataclass
class ReasoningBudget:
    """Patient-specific acquisition budget for the BAR reasoning loop."""

    total: float = 5.0
    min_budget: float = 1.0
    max_budget: float = 5.0
    query_cost: float = 1.0
    expand_cost: float = 1.0
    select_cost: float = 0.5
    prefilter_cap: int = 8
    evidence_cap: int = 8
    citation_cap: int = 5
    min_edges: int = 1
    fallback_quality_threshold: float = 0.1
    stop_delta: float = 1e-3
    stop_patience: int = 2


@dataclass
class GraphQuery:
    """One executable graph query in the plan-guided navigation step."""

    query_type: str
    anchor_id: Optional[str] = None
    concept_id: Optional[str] = None
    relation: Optional[str] = None
    top_k: int = 8

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GraphQuery":
        return cls(
            query_type=str(data.get("query_type", "")),
            anchor_id=data.get("anchor_id"),
            concept_id=data.get("concept_id"),
            relation=data.get("relation"),
            top_k=int(data.get("top_k", 8)),
        )


@dataclass
class QueryResult:
    valid: bool
    candidate_edge_ids: List[str] = field(default_factory=list)
    terminal: bool = False
    payload: JsonDict = field(default_factory=dict)
    cost: float = 0.0
    message: str = ""


@dataclass
class ReasoningStep:
    step_index: int
    hypothesis: str
    query: GraphQuery
    expected_evidence: str = ""
    budget_fraction: float = 0.0
    candidate_edge_ids: List[str] = field(default_factory=list)
    selected_edge_ids: List[str] = field(default_factory=list)
    verified: bool = False
    revision_reason: str = ""


@dataclass
class ReasoningTrace:
    sample_id: str
    target_disease_id: str
    horizon_days: int
    patient_budget: float
    steps: List[ReasoningStep] = field(default_factory=list)
    evidence_edge_ids: List[str] = field(default_factory=list)
    citation_edge_ids: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    utility_history: List[float] = field(default_factory=list)
    fallback: bool = False
    terminal_reason: str = ""

    @property
    def reasoning_steps(self) -> int:
        return len(self.steps)

    def add_step(self, step: ReasoningStep, cost: float) -> None:
        self.steps.append(step)
        self.total_cost += cost


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output
