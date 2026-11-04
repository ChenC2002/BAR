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


def assert_match(actual, expected, path):
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        for key in actual:
            assert_match(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            assert_match(value, reference, f"{path}[{index}]")
        return
    if (
        isinstance(actual, float)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
        ):
            return
    elif type(actual) is type(expected) and actual == expected:
        return
    raise ValueError(f"Replay mismatch: {path}")


def verify(engine, encoder, public):
    count = 0
    largest_error = 0.0
    for example in public["cases"]:
        for key, recorded in example["outputs"].items():
            horizon, budget = key.split(":")
            actual = predict(engine, encoder, public, example, int(horizon), budget)
            error = abs(actual.probability - recorded["probability"])
            largest_error = max(largest_error, error)
            values = actual.model_dump()
            assert_match(
                values,
                {field: recorded[field] for field in values},
                f"{example['id']} / {key}",
            )
            store = engine.stores[public["target"]]
            ids = {
                eid for step in actual.trace.steps for eid in step.candidates + step.selected
            } | set(actual.trace.citation_ids)
            edges = [store.edges[eid] for eid in sorted(ids)]
            nodes = {
                public["target"],
                *recorded["anchors"],
                *(node for edge in edges for node in (edge.head, edge.tail)),
            }
            assert_match(
                [edge.model_dump() for edge in edges],
                recorded["edges"],
                f"{example['id']} / {key}.edges",
            )
            assert_match(
                [store.nodes[node].model_dump() for node in sorted(nodes)],
                recorded["nodes"],
                f"{example['id']} / {key}.nodes",
            )
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
