"""Run the released checkpoint on the demo examples, or verify all recorded episodes."""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from zipfile import ZipFile

import torch
from mimic_features import FeatureEncoder

from bar.policy import DemoPolicy
from bar.schemas import Case
from bar.training import load_checkpoint


def read_bundle(path):
    with ZipFile(path) as bundle:
        manifest = json.loads(bundle.read("bar-model/manifest.json"))
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported model bundle version")
        files = {}
        for name in ("bar.pt", "features.json", "public_demo.json", "config.json"):
            content = bundle.read("bar-model/" + name)
            if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Checksum mismatch: {name}")
            files[name] = content
    return files


def predict(engine, encoder, public, example, horizon, budget):
    recorded = example["outputs"][f"{horizon}:{budget}"]
    inputs = example["inputs"]
    case = Case(
        sample_id=example["id"],
        disease_id=public["target"],
        horizon_days=horizon,
        ehr_features=encoder.transform([inputs])[0].tolist(),
        anchor_ids=recorded["anchors"],
        diagnosis_codes=inputs["codes"],
        history_codes=inputs["history"],
    )
    original = engine.config
    try:
        if budget != "auto":
            engine.config = original.model_copy(
                update={
                    "reasoning": original.reasoning.model_copy(
                        update={
                            "min_budget": float(budget),
                            "max_budget": float(budget),
                        }
                    ),
                }
            )
        return engine.predict(case)
    finally:
        engine.config = original


def verify(engine, encoder, public):
    count = 0
    largest_error = 0.0
    for example in public["cases"]:
        for key, recorded in example["outputs"].items():
            horizon, budget = key.split(":")
            actual = predict(engine, encoder, public, example, int(horizon), budget)
            error = abs(actual.probability - recorded["probability"])
            largest_error = max(largest_error, error)
            if error > 1e-6:
                raise ValueError(f"Prediction mismatch: {example['id']} / {key}")
            for field in ("evidence_ids", "citation_ids", "fallback", "terminal_reason"):
                if getattr(actual.trace, field) != recorded["trace"][field]:
                    raise ValueError(f"Trace mismatch ({field}): {example['id']} / {key}")
            if len(actual.trace.steps) != len(recorded["trace"]["steps"]):
                raise ValueError(f"Step count mismatch: {example['id']} / {key}")
            for step, expected in zip(actual.trace.steps, recorded["trace"]["steps"], strict=True):
                values = step.model_dump()
                for field in ("query", "candidates", "selected", "accepted", "cost"):
                    if values[field] != expected[field]:
                        raise ValueError(f"Acquisition mismatch ({field}): {example['id']} / {key}")
            for field in ("budget", "total_cost", "frozen_ehr_probability"):
                if not math.isclose(
                    getattr(actual.trace, field), recorded["trace"][field], abs_tol=1e-6
                ):
                    raise ValueError(f"Trace value mismatch ({field}): {example['id']} / {key}")
            count += 1
    return {"verified_episodes": count, "max_probability_error": largest_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("downloads/bar-demo-model.zip"))
    parser.add_argument("--case", default="example-diabetes")
    parser.add_argument("--horizon", type=int, choices=(30, 90, 365), default=90)
    parser.add_argument("--budget", choices=("auto", "2", "6", "12"), default="auto")
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    files = read_bundle(args.bundle)
    public = json.loads(files["public_demo.json"])
    encoder = FeatureEncoder.from_dict(json.loads(files["features.json"]))
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "bar.pt"
        checkpoint.write_bytes(files["bar.pt"])
        engine = load_checkpoint(checkpoint, DemoPolicy())
        if args.verify_all:
            result = verify(engine, encoder, public)
        else:
            example = next((c for c in public["cases"] if c["id"] == args.case), None)
            if example is None:
                parser.error(
                    "unknown example; choose " + ", ".join(c["id"] for c in public["cases"])
                )
            result = predict(
                engine, encoder, public, example, args.horizon, args.budget
            ).model_dump()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
