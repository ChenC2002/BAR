import csv
import importlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("evaluate_mimic_demo")
    runner = importlib.import_module("run_mimic_demo")

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not fit, train, or reconstruct the graph")

    monkeypatch.setattr(module.FeatureEncoder, "fit", forbidden)
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "knowledge", forbidden)
    module.torch.set_num_threads(2)
    patients, admissions, diagnoses, icu, labs = [], [], [], [], []
    for pid in range(20):
        patients.append([pid, "1970-01-01", "M"])
        start = datetime(2020, 1, 1) + timedelta(days=pid)
        end = start + timedelta(days=1)
        admissions.append([pid, pid, start.isoformat(), end.isoformat(), 0])
        diagnoses.append([pid, pid, "4019"])
        icu.append([pid])
        labs.append([pid, pid, "50912", start.isoformat(), 1 + pid / 10, "mg/dL"])
        if pid % 2 == 0:
            admissions.append(
                [
                    pid,
                    100 + pid,
                    (end + timedelta(days=1)).isoformat(),
                    (end + timedelta(days=2)).isoformat(),
                    0,
                ]
            )
            diagnoses.append([pid, 100 + pid, "5859"])
    tables = {
        "PATIENTS.csv": (["SUBJECT_ID", "DOB", "GENDER"], patients),
        "ADMISSIONS.csv": (
            ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"],
            admissions,
        ),
        "DIAGNOSES_ICD.csv": (["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], diagnoses),
        "ICUSTAYS.csv": (["HADM_ID"], icu),
        "LABEVENTS.csv": (
            ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"],
            labs,
        ),
        "mapping.csv": (["ICD9CM", "UMLS"], [["401.9", "C0085580"]]),
    }
    for name, (header, records) in tables.items():
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(records)
    files = module.read_bundle(ROOT / "downloads/bar-demo-model.zip")
    public = json.loads(files["public_demo.json"])
    _, cohort = module.prepare(tmp_path, 20, 7)
    public["summary"]["cohort"] = cohort
    files["public_demo.json"] = json.dumps(public).encode()
    monkeypatch.setattr(module, "read_bundle", lambda path: files)
    return module, files, public


def test_evaluation_recomputes_metrics_with_frozen_checkpoint(evaluation, tmp_path):
    module, files, public = evaluation
    report = module.evaluate_experiment(tmp_path, tmp_path / "mapping.csv", None)
    assert report["reference_check"]["matches"] is False
    assert report["metrics"] != public["summary"]["metrics"]
    assert report["cohort"]["test"]["patients"] == 4
    assert all(row["samples"] == 4 and row["positives"] == 2 for row in report["metrics"])
    assert report["reasoning"]["budget_violations"] == 0
    assert report["checkpoint_sha256"] == module.hashlib.sha256(files["bar.pt"]).hexdigest()
    assert report["macro"]["bar"]["ap"] == pytest.approx(
        sum(row["bar"]["ap"] for row in report["metrics"]) / 3
    )
    assert set(report) == {
        "cohort",
        "metrics",
        "validation_ap",
        "validation_preferred",
        "reasoning",
        "graph",
        "macro",
        "reference_check",
        "checkpoint_sha256",
        "elapsed_seconds",
    }
    for field in ("sample_id", "patient", "labels", "traces", "ehr_features"):
        assert f'"{field}":' not in json.dumps(report)

    # A reference updated to the actual recomputed aggregates must verify on replay.
    for key in ("cohort", "metrics", "validation_ap", "validation_preferred", "reasoning"):
        public["summary"][key] = report[key]
    files["public_demo.json"] = json.dumps(public).encode()
    replay = module.evaluate_experiment(tmp_path, tmp_path / "mapping.csv", None)
    assert replay["reference_check"]["matches"] is True
    assert replay["reference_check"]["mismatches"] == []
    assert replay["metrics"] == report["metrics"]


def test_cohort_mismatch_stops_before_labs_or_predictions(evaluation, monkeypatch, tmp_path):
    module, files, public = evaluation
    public["summary"]["cohort"]["test"]["admissions"] += 1
    files["public_demo.json"] = json.dumps(public).encode()

    def forbidden(*args, **kwargs):
        pytest.fail("Cohort mismatch must stop evaluation before reading laboratories")

    monkeypatch.setattr(module, "add_laboratories", forbidden)
    with pytest.raises(ValueError, match="Cohort differs"):
        module.evaluate_experiment(tmp_path, tmp_path / "mapping.csv", None)


def test_cli_writes_mismatch_report_and_refuses_overwrite(evaluation, monkeypatch, tmp_path):
    module, _, _ = evaluation
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_mimic_demo.py",
            "--data",
            str(tmp_path),
            "--mapping",
            str(tmp_path / "mapping.csv"),
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(module.torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(module.os, "umask", lambda mask: None)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    original = out.read_bytes()
    assert json.loads(original)["reference_check"]["matches"] is False
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert out.read_bytes() == original


def test_mapping_preserves_canonical_unambiguous_joins(evaluation, tmp_path):
    module, _, _ = evaluation
    path = tmp_path / "mapping.csv"
    path.write_text("ICD9CM,UMLS\n401.9,C1\n4019,C2\n250.00,C3\n250.00,C4\nE123.4,C5\n")
    assert module.load_code_mapping(path) == {"4019": "C1", "E1234": "C5"}
