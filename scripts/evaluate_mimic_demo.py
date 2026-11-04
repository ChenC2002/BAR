"""Recompute the demo Experiment metrics from the released checkpoint without training."""

import argparse
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from mimic_features import FeatureEncoder, add_laboratories
from predict_demo import assert_match, read_bundle
from run_mimic_demo import (
    DISEASE,
    HORIZONS,
    evaluate,
    load_code_mapping,
    make_case,
    metric_summary,
    prepare,
    reasoning_summary,
    validation_summary,
)

from bar.policy import DemoPolicy
from bar.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def evaluate_experiment(data, mapping_path, bundle):
    start = time.monotonic()
    files = read_bundle(bundle)
    public = json.loads(files["public_demo.json"])
    reference = public["summary"]
    if public["target"] != DISEASE or reference["policy"] != "Offline deterministic policy":
        raise ValueError("Expected the released CKD checkpoint with its offline policy")
    encoder = FeatureEncoder.from_dict(json.loads(files["features.json"]))
    mapping = load_code_mapping(mapping_path)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "bar.pt"
        checkpoint.write_bytes(files["bar.pt"])
        engine = load_checkpoint(checkpoint, DemoPolicy())
    assert_match(engine.config.model_dump(), json.loads(files["config.json"]), "config")
    assert_match(engine.config.seed, reference["seed"], "seed")
    store = engine.stores[DISEASE]

    logging.info("Rebuilding the recorded patient split from MIMIC-III...")
    # The selected count also recovers capped runs; prepare uses the recorded seed.
    count = sum(reference["cohort"][name]["patients"] for name in ("train", "validation", "test"))
    split, cohort = prepare(data, count, reference["seed"])
    try:
        assert_match(cohort, reference["cohort"], "cohort")
    except ValueError as error:
        raise ValueError(
            f"Cohort differs from the released experiment; check input data. {error}"
        ) from error
    del split["train"]
    logging.info("Reading pre-discharge laboratories for validation and test admissions...")
    add_laboratories(data, split)

    def cases(records):
        features = encoder.transform(records)
        for row, values in zip(records, features, strict=True):
            for horizon in HORIZONS:
                yield make_case(row, horizon, values.tolist(), store, mapping)

    logging.info("Evaluating validation: %s admissions x 3 horizons...", len(split["validation"]))
    validation_ap, preferred = validation_summary(evaluate(engine, cases(split["validation"])))
    logging.info("Evaluating test: %s admissions x 3 horizons...", len(split["test"]))
    testing = evaluate(engine, cases(split["test"]))
    metrics = metric_summary(testing)
    summary = {
        "cohort": cohort,
        "metrics": metrics,
        "validation_ap": validation_ap,
        "validation_preferred": preferred,
        "reasoning": reasoning_summary([t for row in testing.values() for t in row["traces"]]),
        "graph": {
            "source": "Local UMLS export",
            "nodes": len(store.nodes),
            "edges": len(store.edges),
            "support_threshold": engine.config.support_threshold,
        },
    }
    mismatches = []
    for key, value in summary.items():
        try:
            assert_match(value, reference[key], key)
        except ValueError as error:
            mismatches.append(str(error))
    return {
        **summary,
        "macro": {
            model: {
                metric: float(np.mean([row[model][metric] for row in metrics]))
                for metric in ("auroc", "ap", "brier")
            }
            for model in ("bar", "ehr")
        },
        "reference_check": {
            "matches": not mismatches,
            "absolute_tolerance": 1e-6,
            "mismatches": mismatches,
        },
        "checkpoint_sha256": hashlib.sha256(files["bar.pt"]).hexdigest(),
        "elapsed_seconds": time.monotonic() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory of MIMIC-III CSV files")
    parser.add_argument(
        "--mapping", type=Path, required=True, help="ICD9CM_to_UMLS.csv used by the run"
    )
    parser.add_argument("--bundle", type=Path, default=ROOT / "downloads/bar-demo-model.zip")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ckd-evaluation.json"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new --out path")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.umask(0o077)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    try:
        report = evaluate_experiment(args.data, args.mapping, args.bundle)
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(content)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(content, end="")
    if not report["reference_check"]["matches"]:
        parser.exit(1, f"Results differ from the released experiment; see {args.out}\n")
    logging.info("Experiment metrics match the release. Saved %s", args.out)


if __name__ == "__main__":
    main()
