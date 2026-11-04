"""Minimal three-stage training with PyTorch optimizers and on-policy query routing."""

import logging
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


def macro_average_precision(cases: list[TrainingCase], scores: list[float]) -> float:
    from sklearn.metrics import average_precision_score

    grouped = defaultdict(lambda: ([], []))
    for case, score in zip(cases, scores, strict=True):
        grouped[case.task][0].append(case.label)
        grouped[case.task][1].append(score)
    values = [average_precision_score(y, p) for y, p in grouped.values() if 0 < sum(y) < len(y)]
    if not values:
        raise ValueError("validation requires at least one task with both classes")
    return float(sum(values) / len(values))


def train(
    cases: list[TrainingCase],
    stores: dict[str, GraphStore],
    policy: Policy,
    config: Config,
    validation_cases: list[TrainingCase] | None = None,
) -> tuple[BAR, list[dict]]:
    """Train on shuffled minibatches; sample fresh reasoning episodes each round."""
    if not cases:
        raise ValueError("training cases must not be empty")
    if validation_cases is not None:
        if not validation_cases:
            raise ValueError("validation cases must not be empty")
        if {c.sample_id for c in cases} & {c.sample_id for c in validation_cases}:
            raise ValueError("training and validation sample IDs must be disjoint")
    for case in cases:
        if case.disease_id not in stores or case.disease_id not in stores[case.disease_id].nodes:
            raise ValueError(f"missing disease graph for {case.disease_id}")
        if len(case.ehr_features) != len(cases[0].ehr_features):
            raise ValueError("all cases must use the same feature layout")
    torch.manual_seed(config.seed)
    model = make_model(stores, cases, config.hidden_dim, config.beta)
    router = QueryRouter(config.hidden_dim)
    cfg = config.training
    predictor_optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.predictor_lr, weight_decay=cfg.weight_decay
    )
    policy_optimizer = torch.optim.AdamW(router.parameters(), lr=cfg.policy_lr)
    generator = torch.Generator().manual_seed(config.seed)

    def batches(limit=None):
        order = torch.randperm(len(cases), generator=generator).tolist()
        if limit is not None:
            order = order[:limit]
        for start in range(0, len(order), cfg.batch_size):
            yield [cases[i] for i in order[start : start + cfg.batch_size]]

    history = []
    best_score, best_model, stale = float("-inf"), None, 0
    if validation_cases:
        with torch.no_grad():
            for task, idx in model.task_index.items():
                labels = [c.label for c in cases if c.task == task]
                prevalence = max(1e-5, min(1 - 1e-5, sum(labels) / len(labels)))
                model.bias.weight[idx].fill_(torch.logit(torch.tensor(prevalence)).item())
    for epoch in range(cfg.baseline_epochs):
        model.train()
        total_loss = 0.0
        for batch in batches():
            labels = torch.tensor([float(c.label) for c in batch], device=model.device)
            loss = F.binary_cross_entropy_with_logits(model.raw_logits(batch), labels)
            predictor_optimizer.zero_grad()
            loss.backward()
            predictor_optimizer.step()
            total_loss += loss.detach().item() * len(batch)
        history.append({"stage": "ehr_only", "epoch": epoch + 1, "loss": total_loss / len(cases)})
        if validation_cases:
            model.eval()
            with torch.no_grad():
                scores = torch.cat(
                    [
                        model.raw_logits(validation_cases[i : i + cfg.batch_size]).sigmoid()
                        for i in range(0, len(validation_cases), cfg.batch_size)
                    ]
                ).tolist()
            score = macro_average_precision(validation_cases, scores)
            history[-1]["validation_ap"] = score
            logging.getLogger(__name__).info("EHR epoch %d validation AP %.6f", epoch + 1, score)
            if score > best_score:
                best_score, best_model, stale = score, deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= cfg.patience:
                break
    if best_model is not None:
        model.load_state_dict(best_model)
    # Fine-tuning starts from the selected EHR snapshot with fresh optimizer moments.
    predictor_optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.finetune_lr or cfg.predictor_lr, weight_decay=cfg.weight_decay
    )
    baseline = deepcopy(model).eval().requires_grad_(False)
    predictions = defaultdict(list)
    with torch.no_grad():
        for case in cases:
            predictions[case.task].append(baseline.raw_logit(case).sigmoid().item())
    budgets = BudgetAllocator.fit(dict(predictions))
    engine = BAR(model, baseline, router, stores, budgets, policy, config)
    reward_baseline = 0.0
    best_reasoning_score, best_reasoning = float("-inf"), None
    for stage, epochs, mode in [
        ("warmup", cfg.warmup_epochs, "uniform"),
        ("reinforce", cfg.policy_epochs, "sample"),
    ]:
        for epoch in range(epochs):
            # eval disables dropout but leaves the supervised computation graph intact.
            model.eval()
            totals = dict(loss=0.0, reward=0.0, policy_loss=0.0)
            count = 0
            for batch in batches(cfg.episodes_per_round):
                losses, rewards, log_sums = [], [], []
                for case in batch:
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
                count += len(batch)
                totals["loss"] += loss.detach().item() * len(batch)
                totals["reward"] += rewards_tensor.mean().item() * len(batch)
                totals["policy_loss"] += policy_loss.detach().item() * len(batch)
            history.append(
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "episodes": count,
                    **{name: value / count for name, value in totals.items()},
                }
            )
            if validation_cases:
                scores = [engine.predict(case).probability for case in validation_cases]
                score = macro_average_precision(validation_cases, scores)
                history[-1]["validation_ap"] = score
                logging.getLogger(__name__).info(
                    "%s epoch %d validation AP %.6f", stage, epoch + 1, score
                )
                if score > best_reasoning_score:
                    best_reasoning_score = score
                    best_reasoning = (deepcopy(model.state_dict()), deepcopy(router.state_dict()))
    if best_reasoning is not None:
        model.load_state_dict(best_reasoning[0])
        router.load_state_dict(best_reasoning[1])
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
