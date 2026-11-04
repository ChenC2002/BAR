"""Package a verified CKD evaluation with its released checkpoint and per-run CSVs."""

import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from package_demo import RELEASE_FILES, archive
from predict_demo import assert_match, read_bundle

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_KEYS = (
    "cohort",
    "metrics",
    "validation_ap",
    "validation_preferred",
    "reasoning",
    "graph",
)


def csv_bytes(records):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return handle.getvalue().encode()


def build(report_path, bundle_path, destination):
    if destination.exists():
        raise FileExistsError("Output already exists; choose a new --out path")
    files = read_bundle(bundle_path)
    reference = json.loads(files["public_demo.json"])["summary"]
    config = json.loads(files["config.json"])
    report = json.loads(report_path.read_text())
    expected_keys = {
        *SUMMARY_KEYS,
        "macro",
        "reference_check",
        "checkpoint_sha256",
        "elapsed_seconds",
    }
    if set(report) != expected_keys:
        raise ValueError("Expected only the aggregate report from evaluate_mimic_demo.py")
    assert_match(config["seed"], reference["seed"], "seed")
    assert_match(
        report["checkpoint_sha256"], hashlib.sha256(files["bar.pt"]).hexdigest(), "checkpoint"
    )
    for key in SUMMARY_KEYS:
        assert_match(report[key], reference[key], key)
    expected_macro = {
        model: {
            metric: mean(row[model][metric] for row in report["metrics"])
            for metric in ("auroc", "ap", "brier")
        }
        for model in ("bar", "ehr")
    }
    assert_match(report["macro"], expected_macro, "macro")
    assert_match(
        report["reference_check"],
        {"matches": True, "absolute_tolerance": 1e-6, "mismatches": []},
        "reference_check",
    )
    elapsed = report["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid elapsed_seconds")
    # Only the released inference-bundle members may be included in a public attachment.
    with ZipFile(bundle_path) as bundle:
        allowed = {*files, *RELEASE_FILES, "README.txt", "manifest.json"}
        if sorted(bundle.namelist()) != sorted(f"bar-model/{name}" for name in allowed):
            raise ValueError("Unexpected model bundle members")
        manifest = json.loads(bundle.read("bar-model/manifest.json"))
        for name in allowed - {"manifest.json"}:
            digest = hashlib.sha256(bundle.read(f"bar-model/{name}")).hexdigest()
            if digest != manifest["files"][name]:
                raise ValueError(f"Checksum mismatch: {name}")

    seed = config["seed"]
    identity = {"run_id": f"seed-{seed}", "seed": seed, "dataset": "MIMIC-III", "disease": "CKD"}
    per_run = [
        {
            **identity,
            "model": model,
            "horizon_days": row["horizon"],
            "samples": row["samples"],
            "positives": row["positives"],
            "prevalence": row["prevalence"],
            **row[model],
        }
        for row in report["metrics"]
        for model in ("bar", "ehr")
    ]
    macro = [{**identity, "model": model, **report["macro"][model]} for model in ("bar", "ehr")]
    payload = {
        "bar-demo-model.zip": bundle_path.read_bytes(),
        f"runs/seed-{seed}/evaluation.json": (
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        ).encode(),
        "per_run.csv": csv_bytes(per_run),
        "macro.csv": csv_bytes(macro),
        **{name: (ROOT / name).read_bytes() for name in RELEASE_FILES},
        "README.md": (
            "# CKD Demo Results\n\n"
            f"One trained run (seed {seed}) under the MIMIC-III demo configuration.\n"
            "BAR and its frozen Stage-1 EHR baseline are evaluated at 30, 90 and 365 days.\n"
            "The three horizons are tasks within one run, not three independent runs.\n\n"
            "- `per_run.csv`: test AUROC, average precision (AP), Brier score and cohort counts.\n"
            "- `macro.csv`: unweighted mean of each metric across the three horizons.\n"
            f"- `runs/seed-{seed}/evaluation.json`: the actual checkpoint re-evaluation report.\n"
            "- `bar-demo-model.zip`: trained weights, frozen EHR baseline, router, evidence graph, "
            "uncertainty distributions, fitted preprocessing and configuration.\n"
            "- `manifest.json`: SHA-256 checksums of the attachment files.\n\n"
            "Metrics are fractions, not percentages. AP is computed with scikit-learn's "
            "average_precision_score. `ehr` denotes the frozen EHR baseline.\n\n"
            "## Recompute\n\n"
            "From a BAR repository checkout, with authorized MIMIC-III data and the original "
            "ICD9CM-to-UMLS mapping, run (substitute the extracted attachment path):\n\n"
            "```bash\n"
            'python -m pip install ".[experiment]"\n'
            "python scripts/evaluate_mimic_demo.py \\\n"
            "  --bundle /path/to/bar-ckd-results/bar-demo-model.zip \\\n"
            "  --data /path/to/MIMICIII/data \\\n"
            "  --mapping /path/to/KG_mapping/ICD9CM_to_UMLS.csv \\\n"
            "  --out artifacts/ckd-evaluation.json\n"
            "python scripts/package_ckd_results.py \\\n"
            "  --report artifacts/ckd-evaluation.json \\\n"
            "  --bundle /path/to/bar-ckd-results/bar-demo-model.zip \\\n"
            "  --out artifacts/bar-ckd-results.zip\n"
            "```\n\n"
            "No retraining or LLM API is required. Evaluation metrics must match the release "
            "within 1e-6; elapsed time varies by machine.\n"
        ).encode(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / "results.zip"
        size = archive(staged, "bar-ckd-results", payload)
        with destination.open("xb") as handle:
            handle.write(staged.read_bytes())
    return {"path": str(destination), "bytes": size, "runs": 1, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "downloads/bar-demo-model.zip")
    parser.add_argument("--out", type=Path, default=Path("artifacts/bar-ckd-results.zip"))
    args = parser.parse_args()
    try:
        result = build(args.report, args.bundle, args.out)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
