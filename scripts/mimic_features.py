"""Discharge-bounded laboratory summaries and training-fitted tabular features."""

import csv
from collections import Counter
from datetime import datetime

import numpy as np

# MIMIC-III D_LABITEMS: central laboratory chemistry/hematology, not blood gases.
LABS = {
    "50912": ("creatinine", "mg/dL"),
    "51006": ("urea_nitrogen", "mg/dL"),
    "50971": ("potassium", "mEq/L"),
    "50983": ("sodium", "mEq/L"),
    "50882": ("bicarbonate", "mEq/L"),
    "51222": ("hemoglobin", "g/dL"),
    "50862": ("albumin", "g/dL"),
    "50931": ("glucose", "mg/dL"),
}


def add_laboratories(root, split):
    records = {r["id"].split(":")[1]: r for rows in split.values() for r in rows}
    intervals = {
        hadm: (datetime.fromisoformat(r["admit"]), datetime.fromisoformat(r["time"]))
        for hadm, r in records.items()
    }
    values = {}
    with (root / "LABEVENTS.csv").open(newline="") as handle:
        for event in csv.DictReader(handle):
            hadm, item = event["HADM_ID"], event["ITEMID"]
            if hadm not in records or item not in LABS or not event["VALUENUM"]:
                continue
            if event["SUBJECT_ID"] != records[hadm]["patient"]:
                raise ValueError("Laboratory admission/patient join mismatch")
            name, unit = LABS[item]
            if event["VALUEUOM"].lower() != unit.lower():
                continue
            when, value = datetime.fromisoformat(event["CHARTTIME"]), float(event["VALUENUM"])
            start, end = intervals[hadm]
            if not start <= when <= end or not np.isfinite(value) or value <= 0:
                continue
            key = hadm, name
            if key not in values:
                values[key] = [when, value, value, value]
            else:
                previous = values[key]
                previous[2], previous[3] = min(previous[2], value), max(previous[3], value)
                if when > previous[0]:
                    previous[0], previous[1] = when, value
    for row in records.values():
        row["labs"] = {}
    for (hadm, name), (_, last, minimum, maximum) in values.items():
        records[hadm]["labs"][name] = [last, minimum, maximum]


def tokens(codes):
    return {token for code in codes for token in ("code:" + code, "group:" + code[:3])}


class FeatureEncoder:
    def fit(self, records):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        counts = Counter(t for r in records for t in tokens(r["codes"] + r["history"]))
        self.vocabulary = [
            t for t, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if count >= 10
        ][:512]
        self.numeric = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        )
        self.numeric.fit([self.numbers(r) for r in records])
        return self

    @staticmethod
    def numbers(row):
        return [
            row.get("age", np.nan),
            row.get("male", np.nan),
            np.log1p(row.get("stay_days", 0)),
            np.log1p(row.get("prior_visits", 0)),
            np.log1p(len(row["codes"])),
            np.log1p(len(row["history"])),
            *(
                value
                for name, _ in LABS.values()
                for value in row.get("labs", {}).get(name, [np.nan] * 3)
            ),
        ]

    def transform(self, records):
        binary = np.asarray(
            [
                [
                    float(t in source)
                    for source in (tokens(r["codes"]), tokens(r["history"]))
                    for t in self.vocabulary
                ]
                for r in records
            ],
            dtype=np.float32,
        )
        numeric = np.clip(self.numeric.transform([self.numbers(r) for r in records]), -8, 8)
        return np.concatenate([binary, numeric], axis=1).astype(np.float32)
