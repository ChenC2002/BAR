"""Validated graph, patient, and LLM contracts. No tensors cross the LLM boundary."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Provenance(Record):
    source_id: Identifier
    title: str = ""
    url: str = ""
    snippet: str = ""
    date: str = ""
    score: Probability = 0.0


class Node(Record):
    node_id: Identifier
    name: str = ""
    node_type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Edge(Record):
    edge_id: Identifier
    head: Identifier
    relation: Identifier
    tail: Identifier
    support_score: Probability
    equiv_group_id: str = ""
    provenance: list[Provenance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.equiv_group_id or self.edge_id


class EvidenceGraph(Record):
    nodes: list[Node]
    edges: list[Edge]
    equiv_groups: dict[str, list[str]] = Field(default_factory=dict)
    expansion_guide: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references(self) -> Self:
        nodes = {n.node_id for n in self.nodes}
        edges = {e.edge_id: e for e in self.edges}
        if len(nodes) != len(self.nodes) or len(edges) != len(self.edges):
            raise ValueError("node and edge identifiers must be unique")
        for edge in self.edges:
            if edge.head not in nodes or edge.tail not in nodes:
                raise ValueError(f"unknown endpoint on edge {edge.edge_id}")
        for node, ids in self.expansion_guide.items():
            if node not in nodes or len(ids) != len(set(ids)):
                raise ValueError(f"invalid expansion guide for {node}")
            for edge_id in ids:
                if edge_id not in edges or node not in (edges[edge_id].head, edges[edge_id].tail):
                    raise ValueError(f"non-incident expansion-guide edge: {edge_id}")
        for group, ids in self.equiv_groups.items():
            if len(ids) != len(set(ids)) or any(
                eid not in edges or edges[eid].group != group for eid in ids
            ):
                raise ValueError(f"inconsistent equivalence group: {group}")
        return self


class Case(Record):
    sample_id: Identifier
    disease_id: Identifier
    horizon_days: int = Field(gt=0)
    ehr_features: list[float] = Field(min_length=1)
    anchor_ids: list[str]
    diagnosis_codes: list[str] = Field(default_factory=list)
    history_codes: list[str] = Field(default_factory=list)

    @property
    def task(self) -> str:
        return f"{self.disease_id}:{self.horizon_days}"


class TrainingCase(Case):
    label: Literal[0, 1]


class PatientVisit(Record):
    patient_id: str
    visit_id: str
    admit_time: str
    discharge_time: str
    diagnosis_codes: list[str]


class CohortSample(Record):
    sample_id: str
    patient_id: str
    index_visit_id: str
    index_time: str
    target_disease_id: str
    window_days: int
    label: Literal[0, 1]
    diagnosis_codes: list[str]
    history_codes: list[str]
    anchor_node_ids: list[str]


class PlanStep(Record):
    hypothesis: str
    expected_evidence: str
    node_ids: list[str] = Field(description="Allowed nodes; empty means any current frontier node.")
    relations: list[str] = Field(
        description="Allowed relations; empty means any incident relation."
    )
    budget_fraction: float = Field(gt=0, le=1)


class Plan(Record):
    steps: list[PlanStep] = Field(max_length=32)

    @model_validator(mode="after")
    def fractions(self) -> Self:
        if sum(s.budget_fraction for s in self.steps) > 1 + 1e-6:
            raise ValueError("planned budget fractions must sum to at most one")
        return self


class Selection(Record):
    edge_ids: list[str]


class Verification(Record):
    passed: bool
    reason: str


class Query(Record):
    kind: Literal["guide", "expand"]
    node_id: str
    relation: str | None = None


class StepTrace(Record):
    hypothesis: str
    query: Query
    candidates: list[str]
    selected: list[str]
    accepted: bool
    reason: str
    cost: float
    utility: float


class Trace(Record):
    sample_id: str
    budget: float
    frozen_ehr_probability: Probability
    steps: list[StepTrace] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    total_cost: float = 0
    fallback: bool = False
    terminal_reason: str = "plan_complete"


class Prediction(Record):
    probability: Probability
    citations: list[Edge]
    trace: Trace
