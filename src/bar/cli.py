"""Command-line helpers for BAR."""

from __future__ import annotations

import argparse
import json
import sys

from bar.kg.graph_store import InMemoryGraphStore
from bar.prediction.predictor import HeuristicBARPredictor
from bar.reasoning.loop import HeuristicReasoningPolicy, PlanNavigateVerifyLoop
from bar.reasoning.query_engine import EvidenceGraphQueryEngine
from bar.schemas import CohortSample, EvidenceGraph, ReasoningBudget
from bar.training.stages import stage_plan_as_dicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-evidence-graph")
    validate.add_argument("--graph", required=True)

    reasoning = subparsers.add_parser("demo-reasoning")
    reasoning.add_argument("--graph", required=True)
    reasoning.add_argument("--anchor", action="append", required=True)
    reasoning.add_argument("--disease", required=True)
    reasoning.add_argument("--horizon-days", type=int, default=90)
    reasoning.add_argument("--ehr-score", type=float, default=0.5)
    reasoning.add_argument("--max-budget", type=float, default=5.0)

    subparsers.add_parser("show-training-stages")
    args = parser.parse_args(argv)

    if args.command == "validate-evidence-graph":
        return _validate_graph(args.graph)
    if args.command == "demo-reasoning":
        return _demo_reasoning(args)
    if args.command == "show-training-stages":
        print(json.dumps(stage_plan_as_dicts(), indent=2))
        return 0
    return 1


def _validate_graph(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        graph = EvidenceGraph.from_dict(json.load(handle))
    errors = graph.validate()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    print(f"valid graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return 0


def _demo_reasoning(args: argparse.Namespace) -> int:
    store = InMemoryGraphStore.load(args.graph)
    budget = ReasoningBudget(max_budget=args.max_budget, total=args.max_budget)
    query_engine = EvidenceGraphQueryEngine(store, budget)
    sample = CohortSample(
        sample_id="demo",
        patient_id="demo",
        index_visit_id="demo",
        index_time="",
        target_disease_id=args.disease,
        window_days=args.horizon_days,
        label=0,
        diagnosis_codes=[],
        history_codes=[],
        anchor_node_ids=args.anchor,
    )
    loop = PlanNavigateVerifyLoop(
        store=store,
        query_engine=query_engine,
        budget=budget,
        policy=HeuristicReasoningPolicy(),
    )
    trace = loop.run(sample, ehr_only_score=args.ehr_score)
    prediction = HeuristicBARPredictor().predict(sample, trace, store)
    print(json.dumps({
        "terminal_reason": trace.terminal_reason,
        "patient_budget": trace.patient_budget,
        "total_cost": trace.total_cost,
        "reasoning_steps": trace.reasoning_steps,
        "fallback": trace.fallback,
        "evidence_edge_ids": trace.evidence_edge_ids,
        "citation_edge_ids": trace.citation_edge_ids,
        "risk_probability": prediction.probability,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
