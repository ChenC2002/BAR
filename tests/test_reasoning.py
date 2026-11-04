import pytest
import torch

from bar.budget import BudgetAllocator
from bar.config import ReasoningConfig
from bar.policy import DemoPolicy, PolicyError
from bar.schemas import Plan, Selection, TrainingCase, Verification


def test_budget_uses_task_training_percentiles():
    allocator = BudgetAllocator.fit({"a:30": [0, 0.25, 0.5], "a:90": [0.5, 0.5]})
    cfg = ReasoningConfig()
    assert allocator.allocate("a:30", 0.25, cfg) == pytest.approx(2 + 10 * 2 / 3)
    assert allocator.allocate("a:90", 0.25, cfg) == 2
    assert allocator.allocate("a:90", 0.5, cfg) == 12
    with pytest.raises(ValueError, match="no training"):
        allocator.allocate("unknown", 0.5, cfg)


def test_multihop_updates_state_and_charges_queries_separately(engine, case, store):
    initial = engine.model.initial_state(engine.model.encode(case), case)
    rollout = engine.rollout(case)
    trace = rollout.trace
    assert [s.query.kind for s in trace.steps] == ["guide", "expand", "expand"]
    assert [s.cost for s in trace.steps] == [1.5, 1.0, 1.0]
    assert trace.total_cost == 3.5
    assert not torch.equal(initial, rollout.state)
    assert len(trace.citation_ids) == len(set(trace.citation_ids))
    frontier = set(case.anchor_ids)
    for step in trace.steps:
        assert step.query.node_id in frontier
        assert set(step.selected) <= set(step.candidates)
        if step.accepted:
            frontier.update(
                n for eid in step.selected for n in (store.edges[eid].head, store.edges[eid].tail)
            )


class EmptyPolicy(DemoPolicy):
    def complete(self, operation, payload, schema):
        if operation == "select":
            return Selection(edge_ids=[])
        return super().complete(operation, payload, schema)


def test_empty_selection_charged_and_exact_frozen_fallback(engine, case):
    engine.policy = EmptyPolicy()
    initial = engine.model.initial_state(engine.model.encode(case), case)
    rollout = engine.rollout(case)
    assert rollout.trace.total_cost == 1.5
    assert rollout.trace.evidence_ids == []
    torch.testing.assert_close(rollout.state, initial)
    prediction = engine.predict(case)
    assert prediction.probability == prediction.trace.frozen_ehr_probability
    assert prediction.citations == []


class BadSelection(DemoPolicy):
    def __init__(self, ids):
        self.ids = ids
        self.verified = False

    def complete(self, operation, payload, schema):
        if operation == "select":
            return Selection(edge_ids=self.ids)
        if operation == "verify":
            self.verified = True
        return super().complete(operation, payload, schema)


@pytest.mark.parametrize("ids", [["nonexistent"], ["e_htn_ckd"], ["e_diabetes_renal"] * 2])
def test_unknown_unpresented_or_duplicate_edges_fail_before_llm(engine, case, ids):
    engine.policy = BadSelection(ids)
    prediction = engine.predict(case)
    assert prediction.trace.fallback
    assert not engine.policy.verified
    assert prediction.trace.total_cost == 1.5
    assert prediction.trace.steps[0].reason == "inadmissible_selection"


def test_failed_semantics_preserve_state_and_evidence(engine, case):
    class Reject(DemoPolicy):
        def complete(self, operation, payload, schema):
            if operation == "verify":
                return Verification(passed=False, reason="contradiction")
            return super().complete(operation, payload, schema)

    engine.policy = Reject()
    initial = engine.model.initial_state(engine.model.encode(case), case)
    rollout = engine.rollout(case)
    torch.testing.assert_close(rollout.state, initial)
    assert rollout.trace.evidence_ids == []
    assert rollout.trace.total_cost == 1.5


def test_revision_replaces_remaining_plan_preserving_accepted_steps(engine, case):
    class Revise(DemoPolicy):
        def __init__(self):
            self.selections = 0

        def complete(self, operation, payload, schema):
            if operation == "plan":
                plan = super().complete(operation, payload, schema)
                return Plan(
                    steps=[
                        plan.steps[0].model_copy(update={"budget_fraction": 0.2}) for _ in range(5)
                    ]
                )
            if operation == "select":
                self.selections += 1
                if self.selections == 2:
                    return Selection(edge_ids=[])
            if operation == "revise":
                assert len(payload["accepted"]) == 1
                assert payload["feedback"]["selected"] == []
                plan = super().complete("plan", payload, schema)
                return Plan(steps=plan.steps[:1])
            return super().complete(operation, payload, schema)

    engine.policy = Revise()
    trace = engine.rollout(case).trace
    assert len(trace.steps) == 3
    assert [s.accepted for s in trace.steps] == [True, False, True]
    assert len(trace.evidence_ids) == 2


def test_budget_and_capacity_are_hard_limits(engine, case):
    engine.config.reasoning = ReasoningConfig(
        min_budget=2, max_budget=2, min_quality=0, fallback_quality=0
    )
    trace = engine.rollout(case).trace
    assert trace.total_cost == 1.5 <= trace.budget
    assert trace.terminal_reason == "budget_exhausted"
    engine.config.reasoning = ReasoningConfig(
        min_budget=12, max_budget=12, evidence_cap=1, min_quality=0, fallback_quality=0
    )
    trace = engine.rollout(case).trace
    assert len(trace.evidence_ids) == 1
    assert trace.terminal_reason == "evidence_cap"


def test_policy_failure_cost_and_fallback(engine, case):
    class Broken(DemoPolicy):
        def complete(self, operation, payload, schema):
            if operation == "select":
                raise PolicyError("refusal")
            return super().complete(operation, payload, schema)

    engine.policy = Broken()
    prediction = engine.predict(case)
    assert prediction.trace.total_cost == 1.5
    assert prediction.trace.terminal_reason == "policy_error"
    assert prediction.probability == prediction.trace.frozen_ehr_probability


def test_labels_never_enter_policy_context(engine, case):
    class Spy(DemoPolicy):
        def complete(self, operation, payload, schema):
            assert "label" not in str(payload)
            assert "ehr_features" not in payload
            return super().complete(operation, payload, schema)

    engine.policy = Spy()
    engine.rollout(TrainingCase(**case.model_dump(), label=1))


def test_empty_anchors_cost_nothing(engine, case):
    prediction = engine.predict(case.model_copy(update={"anchor_ids": []}))
    assert prediction.trace.total_cost == 0
    assert prediction.trace.terminal_reason == "no_anchors"


def test_failed_steps_count_toward_stagnation(engine, case):
    class RepeatFailure(EmptyPolicy):
        def complete(self, operation, payload, schema):
            if operation == "revise":
                return DemoPolicy().complete("plan", payload, schema)
            return super().complete(operation, payload, schema)

    engine.policy = RepeatFailure()
    engine.config.reasoning = ReasoningConfig(min_budget=12, max_budget=12, stop_patience=2)
    trace = engine.rollout(case).trace
    assert len(trace.steps) == 2
    assert trace.total_cost == 2.5
    assert trace.terminal_reason == "quality_stagnation"


def test_low_quality_never_reaches_semantic_verifier(engine, case):
    engine.policy = BadSelection(["e_diabetes_renal"])
    engine.config.reasoning = ReasoningConfig(min_quality=1)
    prediction = engine.predict(case)
    assert prediction.trace.fallback
    assert not engine.policy.verified


def test_plan_cannot_jump_to_undiscovered_nodes(engine, case):
    class Jump(DemoPolicy):
        def complete(self, operation, payload, schema):
            result = super().complete(operation, payload, schema)
            if operation == "plan":
                result.steps[0].node_ids = ["CKD"]
            return result

    engine.policy = Jump()
    trace = engine.rollout(case).trace
    assert trace.terminal_reason == "no_actions"
    assert trace.total_cost == 0


def test_seen_edges_do_not_consume_prefilter_slots(engine, case, monkeypatch):
    engine.config.reasoning.prefilter_cap = 1
    monkeypatch.setattr(
        engine.model, "quality", lambda edge, state: state.new_tensor(edge.support_score)
    )
    trace = engine.rollout(case).trace
    assert len(trace.evidence_ids) == 3
    assert all(step.candidates and step.accepted for step in trace.steps)
    assert trace.total_cost == 3.5


def test_provenance_free_edges_cannot_be_acquired(engine, case, store):
    for edge in store.edges.values():
        edge.provenance.clear()
    prediction = engine.predict(case)
    assert prediction.trace.fallback
    assert prediction.trace.evidence_ids == []
    assert prediction.citations == []
    assert prediction.trace.total_cost == 0


def test_guide_query_respects_plan_relations(engine, case):
    class Restricted(DemoPolicy):
        def complete(self, operation, payload, schema):
            result = super().complete(operation, payload, schema)
            if operation == "plan":
                result.steps[0].relations = ["associated_with"]
            return result

    engine.policy = Restricted()
    trace = engine.rollout(case).trace
    assert trace.terminal_reason == "no_actions"
    assert trace.evidence_ids == []
    assert trace.total_cost == 0
