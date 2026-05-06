import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bar.eval.metrics import auroc, average_precision
from bar.kg.graph_store import InMemoryGraphStore
from bar.prediction.predictor import HeuristicBARPredictor
from bar.reasoning.loop import HeuristicReasoningPolicy, PlanNavigateVerifyLoop
from bar.reasoning.query_engine import EvidenceGraphQueryEngine
from bar.schemas import CohortSample, Edge, EvidenceGraph, Node, Provenance, ReasoningBudget
from bar.training.rewards import binary_cross_entropy, total_reward


def make_graph():
    return EvidenceGraph(
        nodes={
            "DIABETES": Node("DIABETES", "Diabetes", "condition"),
            "CKD": Node("CKD", "Chronic kidney disease", "target_disease"),
            "HTN": Node("HTN", "Hypertension", "condition"),
        },
        edges={
            "e1": Edge(
                "e1",
                "DIABETES",
                "risk_factor_for",
                "CKD",
                0.92,
                "g1",
                [Provenance("demo-curated", score=0.92)],
            ),
            "e2": Edge(
                "e2",
                "HTN",
                "associated_with",
                "CKD",
                0.81,
                "g2",
                [Provenance("demo-text", score=0.81)],
            ),
        },
        equiv_groups={"g1": ["e1"], "g2": ["e2"]},
        expansion_guide={"DIABETES": ["e1"], "HTN": ["e2"]},
    )


def main():
    store = InMemoryGraphStore(make_graph())
    sample = CohortSample(
        sample_id="s1",
        patient_id="p1",
        index_visit_id="v1",
        index_time="2026-01-01T00:00:00",
        target_disease_id="CKD",
        window_days=90,
        label=1,
        diagnosis_codes=["250.00"],
        history_codes=[],
        anchor_node_ids=["DIABETES"],
    )
    budget = ReasoningBudget(max_budget=5.0, min_budget=1.0, fallback_quality_threshold=0.01)
    loop = PlanNavigateVerifyLoop(
        store=store,
        query_engine=EvidenceGraphQueryEngine(store, budget),
        budget=budget,
        policy=HeuristicReasoningPolicy(),
    )
    trace = loop.run(sample, ehr_only_score=0.5)
    assert trace.evidence_edge_ids == ["e1"], trace.evidence_edge_ids
    assert trace.citation_edge_ids == ["e1"], trace.citation_edge_ids

    predictor = HeuristicBARPredictor()
    raw_probability = predictor.raw_predict(sample)
    aug = predictor.predict(sample, trace, store)
    raw_loss = binary_cross_entropy(raw_probability, sample.label)
    aug_loss = binary_cross_entropy(aug.probability, sample.label)
    reward = total_reward(
        raw_loss,
        aug_loss,
        trace,
        store,
        max_budget=budget.max_budget,
        citation_cap=budget.citation_cap,
    )

    assert reward["citation_integrity"] == 1.0
    assert auroc([0, 1], [0.2, aug.probability]) == 1.0
    assert average_precision([0, 1], [0.2, aug.probability]) == 1.0
    print("smoke ok")


if __name__ == "__main__":
    main()
