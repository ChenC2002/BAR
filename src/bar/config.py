"""One validated configuration for graph construction, reasoning, and training."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from bar.schemas import Probability, Record


class GraphConfig(Record):
    hop_limit: int = Field(default=3, ge=1)
    degree_cap: int = Field(default=50, ge=1)
    hub_edge_cap: int = Field(default=10, ge=1)
    guide_size: int = Field(default=20, ge=1)


class ReasoningConfig(Record):
    min_budget: float = Field(default=2, ge=0)
    max_budget: float = Field(default=12, gt=0)
    query_cost: float = Field(default=1, ge=0)
    expand_cost: float = Field(default=0.5, ge=0)
    select_cost: float = Field(default=0.5, ge=0)
    prefilter_cap: int = Field(default=5, ge=1)
    evidence_cap: int = Field(default=8, ge=1)
    citation_cap: int = Field(default=5, ge=1)
    min_edges: int = Field(default=1, ge=1)
    min_quality: Probability = 0.1
    fallback_quality: Probability = 0.1
    stop_delta: float = Field(default=0.001, ge=0)
    stop_patience: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.max_budget < self.min_budget or self.min_edges > self.evidence_cap:
            raise ValueError("inconsistent budget or evidence bounds")
        if min(self.query_cost, self.expand_cost) + self.select_cost <= 0:
            raise ValueError("every acquisition attempt must have positive cost")
        return self


class TrainingConfig(Record):
    baseline_epochs: int = Field(default=10, ge=1)
    warmup_epochs: int = Field(default=3, ge=1)
    policy_epochs: int = Field(default=3, ge=1)
    predictor_lr: float = Field(default=0.001, gt=0)
    policy_lr: float = Field(default=0.001, gt=0)
    gain_weight: float = Field(default=1, ge=0)
    cost_weight: float = Field(default=0.1, ge=0)
    integrity_weight: float = Field(default=0.25, ge=0)
    baseline_decay: float = Field(default=0.99, ge=0, lt=1)


class Config(Record):
    seed: int = 7
    hidden_dim: int = Field(default=32, ge=1)
    beta: float = Field(default=1, gt=0)
    support_threshold: Probability = 0.3
    graph: GraphConfig = Field(default_factory=GraphConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
