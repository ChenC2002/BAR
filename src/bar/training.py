"""Minimal three-stage training with PyTorch optimizers and on-policy query routing."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from bar.budget import BudgetAllocator
from bar.config import Config
from bar.graph import GraphStore
from bar.model import BARModel, QueryRouter, make_model
from bar.policy import Policy
from bar.reasoning import BAR, Rollout
from bar.schemas import EvidenceGraph, TrainingCase


def citation_integrity(ids: list[str], store: GraphStore, config: Config) -> float:
    if (
        not ids
        or len(ids) > config.reasoning.citation_cap
        or any(eid not in store.edges for eid in ids)
    ):
        return 0.0
    edges = [store.edges[eid] for eid in ids]
    return float(
        len({e.group for e in edges}) == len(edges)
        and all(e.support_score >= config.support_threshold and e.provenance for e in edges)
    )


def paired_losses(
    model: BARModel, case: TrainingCase, rollout: Rollout, store: GraphStore
) -> tuple[Tensor, Tensor]:
    """Eq. (9): same current predictor, same case, no dropout, with/without evidence."""
    if model.training:
        raise ValueError("paired losses require eval mode (autograd can remain enabled)")
    evidence = (
        [] if rollout.trace.fallback else [store.edges[eid] for eid in rollout.trace.evidence_ids]
    )
    pooled = model.pool(evidence, rollout.state)
    raw = model.logit(rollout.ehr, torch.zeros_like(pooled), case)
    augmented = model.logit(rollout.ehr, pooled, case)
    label = raw.new_tensor(float(case.label))
    return F.binary_cross_entropy_with_logits(raw, label), F.binary_cross_entropy_with_logits(
        augmented, label
    )


def reinforce_loss(log_sums: Tensor, rewards: Tensor, baseline: float) -> Tensor:
    return -((rewards.detach() - baseline) * log_sums).mean()


def train(
    cases: list[TrainingCase], stores: dict[str, GraphStore], policy: Policy, config: Config
) -> tuple[BAR, list[dict]]:
    """Consume training data only. Small full batches keep this reference implementation readable."""
    if not cases:
        raise ValueError("training cases must not be empty")
    for case in cases:
        if case.disease_id not in stores or case.disease_id not in stores[case.disease_id].nodes:
            raise ValueError(f"missing disease graph for {case.disease_id}")
        if len(case.ehr_features) != len(cases[0].ehr_features):
            raise ValueError("all cases must use the same feature layout")
    torch.manual_seed(config.seed)
    model = make_model(stores, cases, config.hidden_dim, config.beta)
    router = QueryRouter(config.hidden_dim)
    cfg = config.training
    predictor_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.predictor_lr)
    policy_optimizer = torch.optim.AdamW(router.parameters(), lr=cfg.policy_lr)
    labels = torch.tensor([float(c.label) for c in cases])
    history = []
    for epoch in range(cfg.baseline_epochs):
        model.train()
        loss = F.binary_cross_entropy_with_logits(
            torch.stack([model.raw_logit(c) for c in cases]), labels
        )
        predictor_optimizer.zero_grad()
        loss.backward()
        predictor_optimizer.step()
        history.append({"stage": "ehr_only", "epoch": epoch + 1, "loss": loss.detach().item()})
    baseline = deepcopy(model).eval().requires_grad_(False)
    predictions = defaultdict(list)
    with torch.no_grad():
        for case in cases:
            predictions[case.task].append(baseline.raw_logit(case).sigmoid().item())
    budgets = BudgetAllocator.fit(dict(predictions))
    engine = BAR(model, baseline, router, stores, budgets, policy, config)
    reward_baseline = 0.0
    for stage, epochs, mode in [
        ("warmup", cfg.warmup_epochs, "uniform"),
        ("reinforce", cfg.policy_epochs, "sample"),
    ]:
        for epoch in range(epochs):
            # eval disables dropout but leaves the supervised computation graph intact.
            model.eval()
            losses, rewards, log_sums = [], [], []
            for case in cases:
                rollout = engine.rollout(case, mode=mode)
                store = stores[case.disease_id]
                raw, augmented = paired_losses(model, case, rollout, store)
                quality = citation_integrity(rollout.trace.citation_ids, store, config)
                reward = (
                    cfg.gain_weight * (raw - augmented).detach()
                    - cfg.cost_weight * rollout.trace.total_cost / config.reasoning.max_budget
                    + cfg.integrity_weight * quality
                )
                losses.append(augmented)
                rewards.append(reward)
                log_sums.append(
                    torch.stack(rollout.log_probabilities).sum()
                    if rollout.log_probabilities
                    else reward.new_zeros(())
                )
            rewards_tensor = torch.stack(rewards)
            policy_loss = reinforce_loss(torch.stack(log_sums), rewards_tensor, reward_baseline)
            if mode == "sample":
                policy_optimizer.zero_grad()
                if policy_loss.requires_grad:
                    policy_loss.backward()
                    policy_optimizer.step()
                reward_baseline = (
                    cfg.baseline_decay * reward_baseline
                    + (1 - cfg.baseline_decay) * rewards_tensor.mean().item()
                )
            predictor_optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            predictor_optimizer.step()
            history.append(
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "loss": loss.detach().item(),
                    "reward": rewards_tensor.mean().item(),
                    "policy_loss": policy_loss.detach().item(),
                }
            )
    return engine, history


def save_checkpoint(engine: BAR, path: str | Path) -> None:
    """Inference checkpoint includes vocabularies, graph snapshots, frozen baseline and CDFs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_spec": engine.model.spec,
            "model": engine.model.state_dict(),
            "baseline": engine.baseline.state_dict(),
            "router": engine.router.state_dict(),
            "config": engine.config.model_dump(),
            "budgets": engine.budgets.model_dump(),
            "graphs": {d: store.graph.model_dump() for d, store in engine.stores.items()},
        },
        path,
    )


def load_checkpoint(path: str | Path, policy: Policy) -> BAR:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["format_version"] != 1:
        raise ValueError("unsupported checkpoint version")
    config = Config.model_validate(data["config"])
    model, baseline = BARModel(**data["model_spec"]), BARModel(**data["model_spec"])
    router = QueryRouter(config.hidden_dim)
    model.load_state_dict(data["model"])
    baseline.load_state_dict(data["baseline"])
    router.load_state_dict(data["router"])
    baseline.eval().requires_grad_(False)
    stores = {d: GraphStore(EvidenceGraph.model_validate(g)) for d, g in data["graphs"].items()}
    return BAR(
        model.eval(),
        baseline,
        router.eval(),
        stores,
        BudgetAllocator.model_validate(data["budgets"]),
        policy,
        config,
    )
