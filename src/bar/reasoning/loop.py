"""Budget-aware plan-navigate-verify loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from bar.kg.graph_store import InMemoryGraphStore
from bar.reasoning.query_engine import EvidenceGraphQueryEngine
from bar.reasoning.scoring import PatientConditionedScorer
from bar.schemas import CohortSample, GraphQuery, ReasoningBudget, ReasoningStep, ReasoningTrace


@dataclass
class PlanStepDraft:
    hypothesis: str
    query: GraphQuery
    expected_evidence: str
    budget_fraction: float


class ReasoningPolicy(Protocol):
    def plan(self, sample: CohortSample, budget: float) -> List[PlanStepDraft]:
        ...

    def select_edges(self, step: PlanStepDraft, candidates: List[dict]) -> List[str]:
        ...

    def verify(self, step: PlanStepDraft, selected: List[dict]) -> tuple[bool, str]:
        ...

    def revise(self, failed_step: PlanStepDraft, reason: str, remaining_budget: float) -> List[PlanStepDraft]:
        ...


class HeuristicReasoningPolicy:
    """Small deterministic policy for smoke tests and local demos.

    The research runs should replace this with the configured LLM policy using
    the prompt templates in ``bar.reasoning.prompts``.
    """

    def plan(self, sample: CohortSample, budget: float) -> List[PlanStepDraft]:
        steps = []
        anchors = sample.anchor_node_ids or [sample.target_disease_id]
        share = 1.0 / max(len(anchors), 1)
        for anchor in anchors:
            steps.append(
                PlanStepDraft(
                    hypothesis=f"{anchor} may connect to {sample.target_disease_id}",
                    query=GraphQuery(query_type="QUERY_GUIDE", anchor_id=anchor, top_k=8),
                    expected_evidence="supported disease pathway edge",
                    budget_fraction=share,
                )
            )
        return steps

    def select_edges(self, step: PlanStepDraft, candidates: List[dict]) -> List[str]:
        return [item["edge_id"] for item in candidates[:2]]

    def verify(self, step: PlanStepDraft, selected: List[dict]) -> tuple[bool, str]:
        if not selected:
            return False, "no admissible evidence selected"
        if sum(item["support_score"] for item in selected) / len(selected) < 0.55:
            return False, "selected evidence is below support threshold"
        return True, "supported"

    def revise(self, failed_step: PlanStepDraft, reason: str, remaining_budget: float) -> List[PlanStepDraft]:
        return []


class PlanNavigateVerifyLoop:
    def __init__(
        self,
        store: InMemoryGraphStore,
        query_engine: EvidenceGraphQueryEngine,
        budget: ReasoningBudget,
        policy: ReasoningPolicy,
        scorer: PatientConditionedScorer | None = None,
        support_threshold: float = 0.55,
    ) -> None:
        self.store = store
        self.query_engine = query_engine
        self.budget = budget
        self.policy = policy
        self.scorer = scorer or PatientConditionedScorer()
        self.support_threshold = support_threshold

    def run(self, sample: CohortSample, ehr_only_score: float) -> ReasoningTrace:
        patient_budget = budget_from_ehr_score(
            ehr_only_score,
            min_budget=self.budget.min_budget,
            max_budget=self.budget.max_budget,
        )
        trace = ReasoningTrace(
            sample_id=sample.sample_id,
            target_disease_id=sample.target_disease_id,
            horizon_days=sample.window_days,
            patient_budget=patient_budget,
        )
        plan = self.policy.plan(sample, patient_budget)
        selected_groups = set()
        stagnant_steps = 0
        previous_utility = 0.0

        for step_index, draft in enumerate(plan, start=1):
            if not self._can_afford(trace.total_cost, patient_budget, draft.query.query_type):
                trace.terminal_reason = "budget_exhausted"
                break

            result = self.query_engine.execute(draft.query)
            if not result.valid:
                trace.terminal_reason = "invalid_query"
                break

            candidate_edge_ids = self._prefilter(result.candidate_edge_ids, sample.anchor_node_ids, selected_groups)
            candidate_payload = self.query_engine.candidate_payload(candidate_edge_ids)
            selected_edge_ids = self.policy.select_edges(draft, candidate_payload)
            selected_edge_ids = self._admissible_edges(selected_edge_ids, selected_groups)
            selected_payload = self.query_engine.candidate_payload(selected_edge_ids)
            verified, reason = self.policy.verify(draft, selected_payload)

            step = ReasoningStep(
                step_index=step_index,
                hypothesis=draft.hypothesis,
                query=draft.query,
                expected_evidence=draft.expected_evidence,
                budget_fraction=draft.budget_fraction,
                candidate_edge_ids=candidate_edge_ids,
                selected_edge_ids=selected_edge_ids,
                verified=verified,
                revision_reason="" if verified else reason,
            )

            step_cost = result.cost + (self.budget.select_cost if selected_edge_ids else 0.0)
            trace.add_step(step, step_cost)
            if not verified:
                revisions = self.policy.revise(draft, reason, patient_budget - trace.total_cost)
                plan[step_index:step_index] = revisions
                continue

            for edge_id in selected_edge_ids:
                edge = self.store.edge(edge_id)
                if edge is None:
                    continue
                selected_groups.add(edge.equiv_group_id or edge.edge_id)
                trace.evidence_edge_ids.append(edge_id)

            utility = self._utility(trace.evidence_edge_ids, sample.anchor_node_ids)
            trace.utility_history.append(utility)
            if utility - previous_utility < self.budget.stop_delta:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            previous_utility = utility

            if len(trace.evidence_edge_ids) >= self.budget.evidence_cap:
                trace.terminal_reason = "evidence_cap"
                break
            if stagnant_steps >= self.budget.stop_patience:
                trace.terminal_reason = "quality_stagnation"
                break

        final_utility = trace.utility_history[-1] if trace.utility_history else 0.0
        if len(trace.evidence_edge_ids) < self.budget.min_edges or final_utility < self.budget.fallback_quality_threshold:
            trace.fallback = True
            trace.terminal_reason = trace.terminal_reason or "fallback"
        else:
            trace.terminal_reason = trace.terminal_reason or "stop"
            trace.citation_edge_ids = self._citations(trace.evidence_edge_ids, sample.anchor_node_ids)
        return trace

    def _prefilter(self, edge_ids: List[str], anchor_ids: List[str], selected_groups: set[str]) -> List[str]:
        scored = self.scorer.score_many(edge_ids, self.store, anchor_ids)
        output = []
        for edge_id, _ in scored:
            edge = self.store.edge(edge_id)
            if edge is None:
                continue
            group_id = edge.equiv_group_id or edge.edge_id
            if group_id in selected_groups:
                continue
            output.append(edge_id)
            if len(output) >= self.budget.prefilter_cap:
                break
        return output

    def _admissible_edges(self, edge_ids: List[str], selected_groups: set[str]) -> List[str]:
        output = []
        for edge_id in edge_ids:
            edge = self.store.edge(edge_id)
            if edge is None or edge.support_score < self.support_threshold:
                continue
            group_id = edge.equiv_group_id or edge.edge_id
            if group_id in selected_groups:
                continue
            output.append(edge_id)
        return output

    def _utility(self, edge_ids: List[str], anchor_ids: List[str]) -> float:
        if not edge_ids:
            return 0.0
        scores = [score for _, score in self.scorer.score_many(edge_ids, self.store, anchor_ids)]
        return sum(scores) / max(len(scores), 1)

    def _citations(self, edge_ids: List[str], anchor_ids: List[str]) -> List[str]:
        ranked = [edge_id for edge_id, _ in self.scorer.score_many(edge_ids, self.store, anchor_ids)]
        citations = []
        seen_groups = set()
        for edge_id in ranked:
            edge = self.store.edge(edge_id)
            if edge is None:
                continue
            group_id = edge.equiv_group_id or edge.edge_id
            if group_id in seen_groups:
                continue
            citations.append(edge_id)
            seen_groups.add(group_id)
            if len(citations) >= self.budget.citation_cap:
                break
        return citations

    def _can_afford(self, current_cost: float, patient_budget: float, query_type: str) -> bool:
        min_cost = self.budget.query_cost + self.budget.select_cost
        if query_type == "QUERY_EXPAND":
            min_cost += self.budget.expand_cost
        return current_cost + min_cost <= patient_budget


def budget_from_ehr_score(ehr_only_score: float, min_budget: float, max_budget: float) -> float:
    """B_i = B_min + (B_max - B_min) * uncertainty(y_ehr)."""

    probability = min(max(ehr_only_score, 0.0), 1.0)
    uncertainty = 1.0 - abs(2.0 * probability - 1.0)
    return min_budget + (max_budget - min_budget) * uncertainty
