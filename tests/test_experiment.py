import csv
import importlib.util
import json
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cohort_keeps_no_followup_and_splits_patients_before_horizons(tmp_path):
    patients, admissions, diagnoses, icu = [], [], [], []
    for pid in range(10):
        patients.append([str(pid), "1970-01-01"])
        for visit, start, end, code in (
            (0, "2020-01-01", "2020-01-02", "4019"),
            (1, "2020-01-10", "2020-01-12", "25000"),
            (2, "2020-02-01", "2020-02-02", "5859"),
        ):
            if visit == 2 and pid % 2:
                continue
            hadm = f"{pid}-{visit}"
            admissions.append([str(pid), hadm, start, end, "0"])
            diagnoses.append([str(pid), hadm, code])
            icu.append([hadm])
    tables = {
        "PATIENTS.csv": (["SUBJECT_ID", "DOB"], patients),
        "ADMISSIONS.csv": (
            ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"],
            admissions,
        ),
        "DIAGNOSES_ICD.csv": (["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], diagnoses),
        "ICUSTAYS.csv": (["HADM_ID"], icu),
    }
    for name, (header, records) in tables.items():
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(records)
    split, summary = script("run_mimic_demo").prepare(tmp_path, 10, 7)
    groups = [{row["patient"] for row in records} for records in split.values()]
    assert [len(group) for group in groups] == [7, 1, 2]
    assert not groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2]
    assert summary["eligible_admissions"] == 20
    with pytest.raises(ValueError, match="Empty train/validation/test"):
        script("run_mimic_demo").prepare(tmp_path, 1, 7)
    for records in split.values():
        for row in records:
            assert "5859" not in row["codes"]
            if int(row["patient"]) % 2:
                assert row["labels"] == {"30": 0, "90": 0, "365": 0}
            else:
                assert row["labels"]["30"] == int(row["id"].endswith("-1"))
                assert row["labels"]["90"] == row["labels"]["365"] == 1
            assert row["history"] == (["4019"] if row["id"].endswith("-1") else [])


@pytest.mark.parametrize("limit", [0, -1])
def test_cohort_rejects_nonpositive_patient_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="must be positive"):
        script("run_mimic_demo").prepare(tmp_path, limit, 7)


def test_demo_export_rejects_patient_fields_and_nonexample_ids(tmp_path):
    exporter = script("export_demo")
    source, out = tmp_path / "public.json", tmp_path / "data.js"
    data = {"summary": {}, "cases": [], "target": "CKD", "case_origin": "Example"}
    source.write_text(json.dumps(data))
    exporter.package(source, out)
    assert out.read_text().startswith("window.BAR_DEMO = ")
    data["summary"]["SUBJECT_ID"] = "123"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Patient-level"):
        exporter.package(source, out)
    data["summary"] = {}
    data["cases"] = [{"id": "123", "outputs": {}}]
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manually defined"):
        exporter.package(source, out)


def test_laboratories_exclude_future_other_admissions_and_wrong_units(tmp_path):
    features = script("mimic_features")
    row = {"id": "1:10", "patient": "1", "admit": "2020-01-01", "time": "2020-01-03"}
    with (tmp_path / "LABEVENTS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"])
        writer.writerows(
            [
                ["1", "10", "50912", "2020-01-01", 1.1, "mg/dL"],
                ["1", "10", "50912", "2020-01-03", 1.5, "mg/dL"],
                ["1", "10", "50912", "2020-01-04", 8.0, "mg/dL"],
                ["1", "10", "50912", "2019-12-31", 9.0, "mg/dL"],
                ["1", "11", "50912", "2020-01-02", 10.0, "mg/dL"],
                ["1", "10", "50912", "2020-01-02", 200.0, "umol/L"],
            ]
        )
    features.add_laboratories(tmp_path, {"train": [row]})
    assert row["labs"] == {"creatinine": [1.5, 1.1, 1.5]}


def test_feature_vocabulary_and_imputation_are_fitted_on_training_only():
    pytest.importorskip("sklearn")
    import numpy as np

    features = script("mimic_features")
    train = [{"codes": ["4019"], "history": [], "age": 50 + i} for i in range(12)]
    encoder = features.FeatureEncoder().fit(train)
    before = encoder.numeric[0].statistics_.copy()
    result = encoder.transform([{"codes": ["UNSEEN"], "history": [], "age": 1000}])
    assert "code:UNSEEN" not in encoder.vocabulary
    assert np.isfinite(result).all()
    np.testing.assert_equal(encoder.numeric[0].statistics_, before)


def test_reasoning_summary_counts_actual_budget_violations():
    from types import SimpleNamespace

    summarize = script("run_mimic_demo").reasoning_summary
    traces = [
        SimpleNamespace(budget=2.0, total_cost=1.5, fallback=False),
        SimpleNamespace(budget=6.0, total_cost=6.5, fallback=True),
    ]
    assert summarize(traces) == {
        "mean_budget": 4.0,
        "mean_cost": 4.0,
        "fallback_rate": 0.5,
        "budget_violations": 1,
    }
    with pytest.raises(ValueError, match="empty evaluation"):
        summarize([])


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_single_class_metrics_match_model_selection(labels):
    pytest.importorskip("sklearn")
    metrics = script("run_mimic_demo").measure(labels, [0.2, 0.8])
    assert metrics["ap"] is None
    assert metrics["auroc"] is None
    assert metrics["brier"] == pytest.approx(0.34)


def test_json_preprocessing_matches_fitted_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    import numpy as np

    features = script("mimic_features")
    train = [
        {
            "codes": ["4019"],
            "history": ["25000"] if i % 2 else [],
            "age": 40 + i,
            "male": i % 2,
            "stay_days": i + 1,
            "labs": {"creatinine": [1 + i / 10, 0.7, 2]} if i % 3 else {},
        }
        for i in range(20)
    ]
    fitted = features.FeatureEncoder().fit(train)
    destination = tmp_path / "features.json"
    fitted.save(destination)
    state = json.loads(destination.read_text())
    frozen = features.FeatureEncoder.from_dict(state)
    holdout = train + [{"codes": ["UNSEEN"], "history": []}]
    np.testing.assert_array_equal(fitted.transform(holdout), frozen.transform(holdout))
    state["numeric"]["scale"][0] = 0
    with pytest.raises(ValueError, match="numeric transform"):
        features.FeatureEncoder.from_dict(state)
