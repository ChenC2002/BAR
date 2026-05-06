import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bar.data.cohort import temporal_split_samples
from bar.eval.metrics import auroc, average_precision
from bar.evidence.refinement import SupportScorer
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


def check_config_and_edge_cases():
    with open(os.path.join(ROOT, "configs/bar_default.json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)
    budget = ReasoningBudget(**config["reasoning_loop"])
    assert budget.min_budget == 1.0
    assert budget.max_budget == 5.0

    prior_edge = Edge("prior", "A", "related_to", "B", support_score=0.8)
    assert SupportScorer().score(prior_edge) == 0.8

    same_admission = [
        CohortSample(
            sample_id=f"p1:v1:target{index}:90",
            patient_id="p1",
            index_visit_id="v1",
            index_time="2026-01-01T00:00:00",
            target_disease_id=f"target{index}",
            window_days=90,
            label=0,
            diagnosis_codes=[],
            history_codes=[],
            anchor_node_ids=[],
        )
        for index in range(3)
    ]
    other_admissions = [
        CohortSample(
            sample_id="p2:v1:target:90",
            patient_id="p2",
            index_visit_id="v1",
            index_time="2026-02-01T00:00:00",
            target_disease_id="target",
            window_days=90,
            label=0,
            diagnosis_codes=[],
            history_codes=[],
            anchor_node_ids=[],
        ),
        CohortSample(
            sample_id="p3:v1:target:90",
            patient_id="p3",
            index_visit_id="v1",
            index_time="2026-03-01T00:00:00",
            target_disease_id="target",
            window_days=90,
            label=0,
            diagnosis_codes=[],
            history_codes=[],
            anchor_node_ids=[],
        ),
    ]
    train, valid, test = temporal_split_samples(
        same_admission + other_admissions,
        train_fraction=0.34,
        valid_fraction=0.33,
    )
    split_by_sample = {}
    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        for sample in split:
            split_by_sample[sample.sample_id] = split_name
    admission_splits = {split_by_sample[sample.sample_id] for sample in same_admission}
    assert len(admission_splits) == 1, admission_splits


def main():
    check_config_and_edge_cases()
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
