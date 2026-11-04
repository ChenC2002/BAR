from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from bar.config import Config
from bar.data.cohort import build_first_onset_samples, temporal_split_samples
from bar.graph import refine_graph, support_score
from bar.schemas import CohortSample, Edge, EvidenceGraph, Node, PatientVisit, Provenance


def edge(eid, head, tail, **kwargs):
    return Edge(
        edge_id=eid,
        head=head,
        tail=tail,
        relation="r",
        support_score=0.8,
        provenance=[Provenance(source_id="test")],
        **kwargs,
    )


def test_support_complete_score_and_fixed_weight_components():
    assert support_score(edge("e", "a", "d")) == 0.8
    assert support_score(edge("e", "a", "d", metadata={"n_curated_sources": 3})) == 0.4
    assert support_score(edge("e", "a", "d", metadata={"text_score": 0.8})) == pytest.approx(0.48)
    assert (
        support_score(edge("e", "a", "d", metadata={"n_curated_sources": 0, "text_score": 0})) == 0
    )


def test_refinement_retains_paths_but_not_dangling_cycle():
    graph = EvidenceGraph(
        nodes=[Node(node_id=n) for n in "AXDBYZ"],
        edges=[
            edge("ax", "A", "X"),
            edge("xd", "X", "D"),
            edge("xb", "X", "B"),
            edge("xy", "X", "Y"),
            edge("yz", "Y", "Z"),
            edge("zx", "Z", "X"),
        ],
    )
    refined = refine_graph(graph, "D", ["A"], Config())
    assert {e.edge_id for e in refined.edges} == {"ax", "xd"}
    assert refine_graph(graph, "D", [], Config()).edges == []


@pytest.mark.parametrize("rejection", ["low_support", "no_provenance"])
def test_rejected_edges_do_not_consume_hub_retention_quota(rejection):
    edges = [edge(str(i), "A", "D", metadata={"prior_score": 1 - i / 10}) for i in range(3)]
    for rejected in edges[:2]:
        if rejection == "low_support":
            rejected.support_score = 0.1
        else:
            rejected.provenance.clear()
    graph = EvidenceGraph(nodes=[Node(node_id=n) for n in ("A", "D")], edges=edges)
    config = Config(graph={"degree_cap": 1, "hub_edge_cap": 2})
    assert [e.edge_id for e in refine_graph(graph, "D", ["A"], config).edges] == ["2"]


def test_schema_rejects_bad_graph_references_and_duplicate_ids(store):
    data = store.graph.model_dump()
    data["nodes"].append(data["nodes"][0])
    with pytest.raises(ValidationError, match="unique"):
        EvidenceGraph.model_validate(data)
    data = store.graph.model_dump()
    data["expansion_guide"]["DIABETES"] = ["e_htn_ckd"]
    with pytest.raises(ValidationError, match="non-incident"):
        EvidenceGraph.model_validate(data)


def test_config_rejects_unknown_and_zero_cost_settings():
    with pytest.raises(ValidationError):
        Config.model_validate({"reasoning": {"budget_max": 12}})
    with pytest.raises(ValidationError):
        Config.model_validate({"reasoning": {"query_cost": 0, "expand_cost": 0, "select_cost": 0}})


def test_labels_use_subsequent_discharge_and_exact_window():
    visits = [
        PatientVisit(
            patient_id="p",
            visit_id="1",
            admit_time="2026-01-01",
            discharge_time="2026-01-02",
            diagnosis_codes=["x"],
        ),
        PatientVisit(
            patient_id="p",
            visit_id="2",
            admit_time="2026-01-03",
            discharge_time="2026-02-01T12:00:00",
            diagnosis_codes=["d"],
        ),
    ]
    samples = build_first_onset_samples(visits, {"D": {"d"}}, [30, 31], {"x": "A"})
    assert [s.label for s in samples] == [0, 1]
    assert all(s.index_visit_id == "1" for s in samples)


def test_split_keeps_all_patient_admissions_and_targets_together():
    samples = []
    for patient in range(10):
        for visit in range(2):
            for disease in ("A", "B"):
                samples.append(
                    CohortSample(
                        sample_id=f"{patient}:{visit}:{disease}",
                        patient_id=str(patient),
                        index_visit_id=str(visit),
                        index_time=(
                            datetime(2026, 1, 1) + timedelta(days=patient + 100 * visit)
                        ).isoformat(),
                        target_disease_id=disease,
                        window_days=90,
                        label=0,
                        diagnosis_codes=[],
                        history_codes=[],
                        anchor_node_ids=[],
                    )
                )
    splits = temporal_split_samples(list(reversed(samples)))
    patients = [{s.patient_id for s in split} for split in splits]
    assert [len(p) for p in patients] == [7, 1, 2]
    assert not (patients[0] & patients[1] or patients[1] & patients[2] or patients[0] & patients[2])
    assert sum(map(len, splits)) == len(samples)
