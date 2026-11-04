import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "downloads/bar-ckd-results.zip"
MODEL = ROOT / "downloads/bar-demo-model.zip"
PREFIX = "bar-ckd-results/"


@pytest.fixture
def package(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("package_ckd_results")
    with ZipFile(RELEASE) as bundle:
        content = bundle.read(PREFIX + "runs/seed-7/evaluation.json")
    report = tmp_path / "evaluation.json"
    report.write_bytes(content)
    return module, report


def test_release_csvs_checkpoint_and_checksums():
    with ZipFile(RELEASE) as bundle:
        manifest = json.loads(bundle.read(PREFIX + "manifest.json"))
        assert set(bundle.namelist()) == {
            PREFIX + name for name in [*manifest["files"], "manifest.json"]
        }
        for name, checksum in manifest["files"].items():
            assert hashlib.sha256(bundle.read(PREFIX + name)).hexdigest() == checksum
        assert bundle.read(PREFIX + "bar-demo-model.zip") == MODEL.read_bytes()
        report = json.loads(bundle.read(PREFIX + "runs/seed-7/evaluation.json"))
        rows = list(csv.DictReader(io.StringIO(bundle.read(PREFIX + "per_run.csv").decode())))
        macro = list(csv.DictReader(io.StringIO(bundle.read(PREFIX + "macro.csv").decode())))
        assert len(rows) == 6
        assert {(row["seed"], row["horizon_days"], row["model"]) for row in rows} == {
            ("7", str(h), model) for h in (30, 90, 365) for model in ("bar", "ehr")
        }
        for row in rows:
            expected = next(
                m for m in report["metrics"] if m["horizon"] == int(row["horizon_days"])
            )
            assert row["dataset"] == "MIMIC-III"
            assert row["disease"] == "CKD"
            assert int(row["samples"]) == expected["samples"] == 7487
            assert int(row["positives"]) == expected["positives"]
            for metric in ("auroc", "ap", "brier"):
                assert float(row[metric]) == expected[row["model"]][metric]
        assert len(macro) == 2
        for row in macro:
            for metric in ("auroc", "ap", "brier"):
                assert float(row[metric]) == report["macro"][row["model"]][metric]
        assert b"One trained run (seed 7)" in bundle.read(PREFIX + "README.md")


def test_repack_is_reproducible_and_preserves_existing_output(package, tmp_path):
    module, report = package
    destination = tmp_path / "results.zip"
    metadata = module.build(report, MODEL, destination)
    assert metadata["runs"] == 1
    assert metadata["seed"] == 7
    assert destination.read_bytes() == RELEASE.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        module.build(report, MODEL, destination)
    assert destination.read_bytes() == RELEASE.read_bytes()


@pytest.mark.parametrize("change", ["metric", "macro", "checkpoint", "patient_data", "check"])
def test_rejects_unverified_or_patient_level_results(package, tmp_path, change):
    module, report_path = package
    report = json.loads(report_path.read_text())
    if change == "metric":
        report["metrics"][0]["bar"]["ap"] += 0.1
    elif change == "macro":
        report["macro"]["bar"]["ap"] += 0.1
    elif change == "checkpoint":
        report["checkpoint_sha256"] = "0" * 64
    elif change == "patient_data":
        report["patients"] = [{"SUBJECT_ID": "private"}]
    else:
        report["reference_check"]["matches"] = False
    report_path.write_text(json.dumps(report))
    destination = tmp_path / "results.zip"
    with pytest.raises(ValueError):
        module.build(report_path, MODEL, destination)
    assert not destination.exists()


def test_rejects_extra_model_bundle_members(package, tmp_path):
    module, report = package
    model = tmp_path / "model.zip"
    model.write_bytes(MODEL.read_bytes())
    with ZipFile(model, "a") as bundle:
        bundle.writestr("bar-model/private.json", "{}")
    with pytest.raises(ValueError, match="Unexpected model bundle members"):
        module.build(report, model, tmp_path / "results.zip")
