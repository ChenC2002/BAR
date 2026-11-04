"""Package the experiment's aggregate/example export for the offline viewer."""

import argparse
import json
from pathlib import Path


def package(source, destination):
    data = json.loads(source.read_text())
    if set(data) != {"summary", "cases", "target", "case_origin"}:
        raise ValueError("Expected public_demo.json, not an experiment artifact.")
    for case in data["cases"]:
        if not case["id"].startswith("example-"):
            raise ValueError("Only manually defined examples may be packaged.")
        for output in case["outputs"].values():
            if output["trace"]["sample_id"] != case["id"]:
                raise ValueError("Trace identifier does not match its example.")
    forbidden = {"subject_id", "hadm_id", "patient", "ehr_features", "dob", "admittime"}

    def audit(value):
        if isinstance(value, dict):
            if forbidden.intersection(key.lower() for key in value):
                raise ValueError("Patient-level fields cannot be packaged.")
            for child in value.values():
                audit(child)
        elif isinstance(value, list):
            for child in value:
                audit(child)

    audit(data)
    destination.write_text("window.BAR_DEMO = " + json.dumps(data, allow_nan=False) + ";\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.source, args.out)
