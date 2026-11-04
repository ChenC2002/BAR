"""PyTorch implementation of patient state, edge quality, pooling, and risk heads."""

import torch
from torch import Tensor, nn

from bar.graph import GraphStore
from bar.schemas import Case, Edge, Query


class BARModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_ids: list[str],
        relations: list[str],
        tasks: list[str],
        hidden_dim: int = 32,
        beta: float = 1,
    ):
        super().__init__()
        self.spec = dict(
            input_dim=input_dim,
            node_ids=sorted(set(node_ids)),
            relations=sorted(set(relations)),
            tasks=sorted(set(tasks)),
            hidden_dim=hidden_dim,
            beta=beta,
        )
        self.node_index = {key: i for i, key in enumerate(self.spec["node_ids"])}
        self.relation_index = {key: i for i, key in enumerate(self.spec["relations"])}
        self.task_index = {key: i for i, key in enumerate(self.spec["tasks"])}
        self.beta = beta
        self.ehr_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.nodes = nn.Embedding(max(1, len(self.node_index)), hidden_dim)
        self.relations = nn.Embedding(max(1, len(self.relation_index)), hidden_dim)
        self.targets = nn.Embedding(len(self.task_index), hidden_dim)
        self.state_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU())
        self.transition = nn.GRUCell(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.Embedding(len(self.task_index), hidden_dim)
        self.bias = nn.Embedding(len(self.task_index), 1)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.bias.weight)

    @property
    def device(self):
        return self.nodes.weight.device

    def index(self, value: int) -> Tensor:
        return torch.tensor(value, device=self.device)

    def target(self, case: Case) -> Tensor:
        return self.targets(self.index(self.task_index[case.task]))

    def encode(self, case: Case) -> Tensor:
        if len(case.ehr_features) != self.spec["input_dim"]:
            raise ValueError("ehr_features length does not match model input_dim")
        return self.ehr_encoder(
            torch.tensor(case.ehr_features, dtype=torch.float32, device=self.device)
        )

    def initial_state(self, ehr: Tensor, case: Case) -> Tensor:
        return self.state_projection(torch.cat((ehr, self.target(case))))

    def node(self, node_id: str) -> Tensor:
        return self.nodes(self.index(self.node_index[node_id]))

    def relation(self, relation: str) -> Tensor:
        return self.relations(self.index(self.relation_index[relation]))

    def edge_vector(self, edge: Edge) -> Tensor:
        return self.edge_encoder(
            torch.cat((self.node(edge.head), self.relation(edge.relation), self.node(edge.tail)))
        )

    def quality(self, edge: Edge, state: Tensor) -> Tensor:
        relevance = torch.maximum(state @ self.node(edge.head), state @ self.node(edge.tail))
        return edge.support_score * torch.sigmoid(self.beta * relevance)

    def update(self, state: Tensor, edges: list[Edge]) -> Tensor:
        return self.transition(torch.stack([self.edge_vector(e) for e in edges]).mean(0), state)

    def pool(self, edges: list[Edge], state: Tensor) -> Tensor:
        if not edges:
            return torch.zeros_like(state)
        scores = torch.stack([self.quality(e, state) for e in edges])
        weights = scores / scores.sum().clamp_min(1e-8)
        return (weights[:, None] * torch.stack([self.edge_vector(e) for e in edges])).sum(0)

    def logit(self, ehr: Tensor, evidence: Tensor, case: Case) -> Tensor:
        task = self.index(self.task_index[case.task])
        return self.fusion(torch.cat((ehr, evidence))) @ self.heads(task) + self.bias(task).squeeze(
            -1
        )

    def raw_logit(self, case: Case) -> Tensor:
        ehr = self.encode(case)
        return self.logit(ehr, torch.zeros_like(ehr), case)

    def raw_logits(self, cases: list[Case]) -> Tensor:
        """Vectorized EHR-only forward pass for training and evaluation batches."""
        if not cases:
            raise ValueError("cases must not be empty")
        if any(len(c.ehr_features) != self.spec["input_dim"] for c in cases):
            raise ValueError("ehr_features length does not match model input_dim")
        features = torch.tensor(
            [c.ehr_features for c in cases], dtype=torch.float32, device=self.device
        )
        tasks = torch.tensor([self.task_index[c.task] for c in cases], device=self.device)
        ehr = self.ehr_encoder(features)
        fused = self.fusion(torch.cat((ehr, torch.zeros_like(ehr)), dim=-1))
        return (fused * self.heads(tasks)).sum(-1) + self.bias(tasks).squeeze(-1)


class QueryRouter(nn.Module):
    """Small trainable query policy; LLM weights remain frozen."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def logits(
        self,
        model: BARModel,
        state: Tensor,
        case: Case,
        actions: list[Query],
        remaining_fraction: float,
    ) -> Tensor:
        features = []
        for action in actions:
            relation = (
                model.relation(action.relation) if action.relation else torch.zeros_like(state)
            )
            features.append(
                torch.cat(
                    (
                        state,
                        model.node(action.node_id),
                        relation,
                        model.target(case),
                        state.new_tensor([remaining_fraction]),
                    )
                ).detach()
            )
        return self.network(torch.stack(features)).squeeze(-1)


def make_model(
    stores: dict[str, GraphStore], cases: list[Case], hidden_dim: int, beta: float
) -> BARModel:
    if not cases:
        raise ValueError("training cases must not be empty")
    return BARModel(
        len(cases[0].ehr_features),
        [node for store in stores.values() for node in store.nodes],
        [e.relation for store in stores.values() for e in store.edges.values()],
        [c.task for c in cases],
        hidden_dim,
        beta,
    )
