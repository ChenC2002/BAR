from copy import deepcopy
from pathlib import Path

import pytest
import torch

from bar.budget import BudgetAllocator
from bar.config import Config, ReasoningConfig
from bar.graph import GraphStore
from bar.model import BARModel, QueryRouter
from bar.policy import DemoPolicy
from bar.reasoning import BAR
from bar.schemas import Case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store():
    return GraphStore.load(ROOT / "examples/tiny_evidence_graph.json")


@pytest.fixture
def case():
    return Case.model_validate_json((ROOT / "examples/tiny_case.json").read_text())


@pytest.fixture
def engine(store, case):
    torch.manual_seed(7)
    config = Config(
        reasoning=ReasoningConfig(
            min_budget=12, max_budget=12, min_quality=0, fallback_quality=0, stop_delta=0
        )
    )
    model = BARModel(4, list(store.nodes), [e.relation for e in store.edges.values()], [case.task])
    baseline = deepcopy(model).eval().requires_grad_(False)
    return BAR(
        model.eval(),
        baseline,
        QueryRouter(32),
        {case.disease_id: store},
        BudgetAllocator.fit({case.task: [0.1, 0.5, 0.9]}),
        DemoPolicy(),
        config,
    )
