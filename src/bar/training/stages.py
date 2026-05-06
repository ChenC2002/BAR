"""Named training stages matching the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StagePlan:
    name: str
    objective: str
    inputs: List[str]
    outputs: List[str]
    metrics: List[str]


def default_stage_plan() -> List[StagePlan]:
    return [
        StagePlan(
            name="stage1_ehr_only_baseline",
            objective="Train the EHR-only predictor by setting z_kg=0 and freeze it.",
            inputs=["MIMIC visits", "target disease code sets"],
            outputs=["ehr_only_checkpoint", "y_ehr", "L_raw"],
            metrics=["AUROC", "AUPRC"],
        ),
        StagePlan(
            name="stage2_supervised_warmup",
            objective="Run the reasoning loop with a fixed policy and train the predictor with acquired evidence.",
            inputs=["refined evidence graphs G_d", "cohort samples", "frozen y_ehr"],
            outputs=["warmup_predictor_checkpoint", "reasoning_traces"],
            metrics=["AUPRC", "citation integrity", "budget utilization"],
        ),
        StagePlan(
            name="stage3_reinforce_policy_optimization",
            objective="Optimize the reasoning policy with paired gain, normalized acquisition cost, and citation integrity.",
            inputs=["reasoning traces", "L_raw", "L_aug", "citation lists"],
            outputs=["bar_checkpoint"],
            metrics=["paired gain", "AUPRC", "citation precision", "reasoning steps"],
        ),
    ]


def stage_plan_as_dicts() -> List[Dict[str, object]]:
    return [stage.__dict__ for stage in default_stage_plan()]
