"""Explicit plan-navigate-verify loop with deterministic enforcement of all constraints."""

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Categorical

from bar.budget import BudgetAllocator
from bar.config import Config
from bar.graph import GraphStore
from bar.model import BARModel, QueryRouter
from bar.policy import Policy, PolicyError
from bar.schemas import (
    Case,
    Plan,
    PlanStep,
    Prediction,
    Query,
    Selection,
    StepTrace,
    Trace,
    Verification,
)


@dataclass
class Rollout:
    trace: Trace
    ehr: Tensor
    state: Tensor
    log_probabilities: list[Tensor]


class BAR:
    def __init__(
        self,
        model: BARModel,
        baseline: BARModel,
        router: QueryRouter,
        stores: dict[str, GraphStore],
        budgets: BudgetAllocator,
        policy: Policy,
        config: Config,
    ):
        self.model, self.baseline, self.router = model, baseline, router
        self.stores, self.budgets, self.policy, self.config = stores, budgets, policy, config

    def _context(self, case: Case, store: GraphStore, trace: Trace, frontier: set[str]) -> dict:
        return {
            "patient": {
                "diagnosis_codes": case.diagnosis_codes,
                "history_codes": case.history_codes,
            },
            "disease": case.disease_id,
            "horizon_days": case.horizon_days,
            "anchors": sorted(set(case.anchor_ids) & store.nodes.keys()),
            "frontier": {key: store.nodes[key].name for key in sorted(frontier)},
            "relations": sorted({e.relation for e in store.edges.values()}),
            "accepted": [store.edges[eid].model_dump() for eid in trace.evidence_ids],
            "remaining_budget": trace.budget - trace.total_cost,
        }

    def _actions(
        self, step: PlanStep, frontier: set[str], first: bool, store: GraphStore, seen: set[str]
    ) -> list[Query]:
        nodes = frontier & set(step.node_ids) if step.node_ids else frontier
        actions = []
        for node in sorted(nodes):
            relations = sorted({e.relation for e in store.incident(node)})
            options = (
                [Query(kind="guide", node_id=node)]
                if first
                else [
                    Query(kind="expand", node_id=node, relation=r)
                    for r in relations
                    if not step.relations or r in step.relations
                ]
            )
            for action in options:
                if any(
                    e.support_score >= self.config.support_threshold and e.group not in seen
                    for e in store.query(action)
                ):
                    actions.append(action)
        return actions

    def rollout(
        self, case: Case, mode: Literal["greedy", "uniform", "sample"] = "greedy"
    ) -> Rollout:
        store = self.stores[case.disease_id]
        cfg = self.config.reasoning
        self.baseline.eval()
        with torch.no_grad():
            frozen_probability = self.baseline.raw_logit(case).sigmoid().item()
        trace = Trace(
            sample_id=case.sample_id,
            frozen_ehr_probability=frozen_probability,
            budget=self.budgets.allocate(case.task, frozen_probability, cfg),
        )
        ehr = self.model.encode(case)
        state = self.model.initial_state(ehr, case)
        result = Rollout(trace, ehr, state, [])
        frontier = set(case.anchor_ids) & store.nodes.keys()
        seen = set()
        previous_utility, stagnant = 0.0, 0
        if not frontier:
            trace.terminal_reason, trace.fallback = "no_anchors", True
            return result
        try:
            plan = self.policy.complete("plan", self._context(case, store, trace, frontier), Plan)
            pending = deque(plan.steps)
        except PolicyError:
            trace.terminal_reason, trace.fallback = "policy_error", True
            return result
        while pending:
            first = not trace.steps
            cost = (cfg.query_cost if first else cfg.expand_cost) + cfg.select_cost
            if trace.total_cost + cost > trace.budget + 1e-9:
                trace.terminal_reason = "budget_exhausted"
                break
            step = pending.popleft()
            actions = self._actions(step, frontier, first, store, seen)
            if not actions:
                trace.terminal_reason = "no_actions"
                break
            logits = self.router.logits(
                self.model, state, case, actions, (trace.budget - trace.total_cost) / trace.budget
            )
            distribution = Categorical(
                logits=torch.zeros_like(logits) if mode == "uniform" else logits
            )
            chosen = logits.argmax() if mode == "greedy" else distribution.sample()
            if mode == "sample":
                result.log_probabilities.append(distribution.log_prob(chosen))
            action = actions[int(chosen)]
            ranked = sorted(
                store.query(action),
                key=lambda e: (-self.model.quality(e, state).detach().item(), e.edge_id),
            )
            candidates = [e for e in ranked[: cfg.prefilter_cap] if e.group not in seen]
            payload = self._context(case, store, trace, frontier)
            payload.update(
                step=step.model_dump(),
                candidates=[e.model_dump() for e in candidates],
                slots=cfg.evidence_cap - len(trace.evidence_ids),
            )
            # Query and selection are one reserved attempt, including empty/failed selections.
            trace.total_cost += cost
            selected, accepted, reason = [], False, "empty_selection"
            policy_error = False
            try:
                selection = self.policy.complete("select", payload, Selection)
                selected = selection.edge_ids
                allowed = {e.edge_id: e for e in candidates}
                groups = [allowed[eid].group for eid in selected if eid in allowed]
                admissible = (
                    bool(selected)
                    and len(selected) <= payload["slots"]
                    and all(eid in allowed for eid in selected)
                    and len(groups) == len(set(groups))
                    and all(
                        allowed[eid].support_score >= self.config.support_threshold
                        and self.model.quality(allowed[eid], state).detach().item()
                        >= cfg.min_quality
                        for eid in selected
                        if eid in allowed
                    )
                )
                if admissible:
                    payload["selected"] = [allowed[eid].model_dump() for eid in selected]
                    verification = self.policy.complete("verify", payload, Verification)
                    accepted, reason = verification.passed, verification.reason
                elif selected:
                    reason = "inadmissible_selection"
            except PolicyError:
                policy_error, reason = True, "policy_error"
            if accepted:
                edges = [store.edges[eid] for eid in selected]
                trace.evidence_ids.extend(selected)
                seen.update(e.group for e in edges)
                frontier.update(n for e in edges for n in (e.head, e.tail))
                state = self.model.update(state, edges)
            utility = sum(
                self.model.quality(store.edges[eid], state).detach().item()
                for eid in trace.evidence_ids
            ) / max(1, len(trace.evidence_ids))
            trace.steps.append(
                StepTrace(
                    hypothesis=step.hypothesis,
                    query=action,
                    candidates=[e.edge_id for e in candidates],
                    selected=selected,
                    accepted=accepted,
                    reason=reason,
                    cost=cost,
                    utility=utility,
                )
            )
            stagnant = stagnant + 1 if utility - previous_utility < cfg.stop_delta else 0
            previous_utility = utility
            if policy_error:
                trace.terminal_reason = "policy_error"
                break
            if len(trace.evidence_ids) >= cfg.evidence_cap:
                trace.terminal_reason = "evidence_cap"
                break
            if trace.total_cost + cfg.expand_cost + cfg.select_cost > trace.budget + 1e-9:
                trace.terminal_reason = "budget_exhausted"
                break
            if stagnant >= cfg.stop_patience:
                trace.terminal_reason = "quality_stagnation"
                break
            if not accepted:
                context = self._context(case, store, trace, frontier)
                context.update(failed_step=step.model_dump(), feedback=trace.steps[-1].model_dump())
                try:
                    pending = deque(self.policy.complete("revise", context, Plan).steps)
                except PolicyError:
                    trace.terminal_reason = "policy_error"
                    break
        trace.fallback = (
            len(trace.evidence_ids) < cfg.min_edges
            or previous_utility < cfg.fallback_quality
            or trace.terminal_reason == "policy_error"
        )
        if not trace.fallback:
            trace.citation_ids = sorted(
                trace.evidence_ids,
                key=lambda eid: (-self.model.quality(store.edges[eid], state).detach().item(), eid),
            )[: cfg.citation_cap]
        result.state = state
        return result

    @torch.no_grad()
    def predict(self, case: Case) -> Prediction:
        self.model.eval()
        self.router.eval()
        rollout = self.rollout(case)
        trace, store = rollout.trace, self.stores[case.disease_id]
        if trace.fallback:
            probability = trace.frozen_ehr_probability
        else:
            pooled = self.model.pool(
                [store.edges[eid] for eid in trace.evidence_ids], rollout.state
            )
            probability = self.model.logit(rollout.ehr, pooled, case).sigmoid().item()
        return Prediction(
            probability=probability,
            citations=[store.edges[eid] for eid in trace.citation_ids],
            trace=trace,
        )
