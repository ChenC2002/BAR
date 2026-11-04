from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter

from bar.config import Config, ReasoningConfig, TrainingConfig
from bar.model import QueryRouter, make_model
from bar.policy import DemoPolicy
from bar.schemas import TrainingCase
from bar.training import (
    citation_integrity,
    load_checkpoint,
    paired_losses,
    reinforce_loss,
    save_checkpoint,
    train,
)

ROOT = Path(__file__).resolve().parents[1]


def test_quality_weighted_pool_is_a_vector_average(engine, store, case):
    state = engine.model.initial_state(engine.model.encode(case), case)
    edges = list(store.edges.values())[:2]
    scores = torch.stack([engine.model.quality(e, state) for e in edges])
    expected = sum(s * engine.model.edge_vector(e) for s, e in zip(scores, edges)) / scores.sum()
    torch.testing.assert_close(engine.model.pool(edges, state), expected)


def test_paired_gain_uses_current_snapshot_not_frozen_baseline(engine, store, case):
    case = TrainingCase(**case.model_dump(), label=1)
    rollout = engine.rollout(case)
    raw, augmented = paired_losses(engine.model, case, rollout, store)
    with torch.no_grad():
        engine.baseline.bias.weight.add_(20)
    second_raw, second_augmented = paired_losses(engine.model, case, rollout, store)
    torch.testing.assert_close(raw, second_raw)
    torch.testing.assert_close(augmented, second_augmented)
    rollout.trace.fallback = True
    raw, augmented = paired_losses(engine.model, case, rollout, store)
    torch.testing.assert_close(raw, augmented)


def test_policy_gradients_are_isolated_from_model_and_reward(engine, case):
    case = case.model_copy(update={"anchor_ids": ["DIABETES", "HTN"]})
    rollout = engine.rollout(case, mode="sample")
    reward = torch.tensor([1.0], requires_grad=True)
    loss = reinforce_loss(torch.stack(rollout.log_probabilities).sum().reshape(1), reward, 0.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in engine.router.parameters())
    assert all(p.grad is None for p in engine.model.parameters())
    assert reward.grad is None


def test_integrity_is_binary_and_checks_every_citation(engine, store):
    config = engine.config
    assert citation_integrity(["e_diabetes_renal"], store, config) == 1
    assert citation_integrity([], store, config) == 0
    assert citation_integrity(["e_diabetes_renal", "missing"], store, config) == 0
    assert citation_integrity(["e_diabetes_renal"] * 2, store, config) == 0
    store.edges["e_renal_ckd"].provenance.clear()
    assert citation_integrity(["e_diabetes_renal", "e_renal_ckd"], store, config) == 0


def test_three_training_stages_change_weights_and_checkpoint_roundtrip(store, case, tmp_path):
    cases = TypeAdapter(list[TrainingCase]).validate_json(
        (ROOT / "examples/tiny_cases.json").read_text()
    )
    cfg = Config(
        training=TrainingConfig(baseline_epochs=2, warmup_epochs=1, policy_epochs=2),
        reasoning=ReasoningConfig(min_quality=0, fallback_quality=0),
    )
    torch.manual_seed(cfg.seed)
    initial_model = make_model({"CKD": store}, cases, cfg.hidden_dim, cfg.beta)
    initial_router = QueryRouter(cfg.hidden_dim)
    engine, history = train(cases, {"CKD": store}, DemoPolicy(), cfg)
    assert {item["stage"] for item in history} == {"ehr_only", "warmup", "reinforce"}
    assert any(
        not torch.equal(a, b) for a, b in zip(initial_model.parameters(), engine.model.parameters())
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(initial_router.parameters(), engine.router.parameters())
    )
    assert all(not p.requires_grad and p.grad is None for p in engine.baseline.parameters())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(engine.baseline.parameters(), engine.model.parameters())
    )
    expected_cdf = sorted(
        1 - abs(2 * engine.baseline.raw_logit(c).sigmoid().item() - 1) for c in cases
    )
    assert engine.budgets.uncertainties[case.task] == expected_cdf
    checkpoint = tmp_path / "bar.pt"
    save_checkpoint(engine, checkpoint)
    loaded = load_checkpoint(checkpoint, DemoPolicy())
    prediction, restored = engine.predict(case), loaded.predict(case)
    assert restored.model_dump() == prediction.model_dump()
    assert restored.trace.total_cost <= restored.trace.budget


def test_multiple_targets_have_distinct_heads_and_cdfs(store, tmp_path):
    cases = TypeAdapter(list[TrainingCase]).validate_json(
        (ROOT / "examples/tiny_cases.json").read_text()
    )
    cases += [
        c.model_copy(update={"horizon_days": 365, "sample_id": c.sample_id + "long"}) for c in cases
    ]
    cfg = Config(training=TrainingConfig(baseline_epochs=1, warmup_epochs=1, policy_epochs=1))
    engine, _ = train(cases, {"CKD": store}, DemoPolicy(), cfg)
    assert set(engine.model.task_index) == {"CKD:90", "CKD:365"}
    assert set(engine.budgets.uncertainties) == {"CKD:90", "CKD:365"}


def test_empty_training_data_rejected(store):
    with pytest.raises(ValueError, match="must not be empty"):
        train([], {"CKD": store}, DemoPolicy(), Config())
