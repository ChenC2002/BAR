"""Count ICU returns and associated hospital diagnoses in MIMIC-III v1.4."""

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

HORIZONS = (30, 90, 365)
CODES = {
    "CKD": (("585",), ()),
    "HF": (("428",), ()),
    "COPD": ((), ("49121", "49122", "49322")),
    "HyperK": ((), ("2767",)),
    "HypoG": ((), ("2510", "2511", "2512")),
    "Stroke": (
        (),
        ("43301", "43311", "43321", "43331", "43381", "43391", "43401", "43411", "43491", "436"),
    ),
    "Sepsis": (("038",), ("0031", "0545", "78552", "99591", "99592")),
    "DVT": (
        ("4534",),
        ("45111", "45119", "45181", "45183", "45382", "45384", "45385", "45386", "45387"),
    ),
}


@dataclass(frozen=True)
class Stay:
    start: datetime
    end: datetime
    hadm: int


def rows(root, name):
    path = root / (name + ".csv")
    if path.exists():
        handle = path.open(encoding="utf-8-sig", newline="")
    else:
        handle = gzip.open(root / (name + ".csv.gz"), "rt", encoding="utf-8-sig", newline="")
    with handle:
        yield from csv.DictReader(handle)


def load_data(root):
    dob = {r["SUBJECT_ID"]: datetime.fromisoformat(r["DOB"]) for r in rows(root, "PATIENTS")}
    admissions = defaultdict(list)
    alive = set()
    for r in rows(root, "ADMISSIONS"):
        if not r["ADMITTIME"] or not r["DISCHTIME"]:
            continue
        hadm = int(r["HADM_ID"])
        admissions[r["SUBJECT_ID"]].append(
            Stay(
                datetime.fromisoformat(r["ADMITTIME"]), datetime.fromisoformat(r["DISCHTIME"]), hadm
            )
        )
        if r["HOSPITAL_EXPIRE_FLAG"] == "0":
            alive.add(hadm)

    icu_by_patient = defaultdict(list)
    for r in rows(root, "ICUSTAYS"):
        hadm = int(r["HADM_ID"])
        if r["INTIME"] and r["OUTTIME"]:
            start = datetime.fromisoformat(r["INTIME"])
            end = datetime.fromisoformat(r["OUTTIME"])
            if start < end:
                icu_by_patient[r["SUBJECT_ID"]].append(Stay(start, end, hadm))
    phenotypes = defaultdict(set)
    for r in rows(root, "DIAGNOSES_ICD"):
        code = r["ICD9_CODE"].strip().replace(".", "")
        if code:
            flags = phenotypes[int(r["HADM_ID"])]
            flags.update(
                d
                for d, (prefixes, exact) in CODES.items()
                if code.startswith(prefixes) or code in exact
            )
    return dob, admissions, alive, phenotypes, icu_by_patient


class Accounting:
    def __init__(self):
        self.n0 = 0
        self.candidate_patients = set()
        self.f0 = [0, 0, 0]
        self.nd = dict.fromkeys(CODES, 0)
        self.fd = {d: [0, 0, 0] for d in CODES}
        self.pd = {d: [0, 0, 0] for d in CODES}
        self.at_year = {
            k: set()
            for k in (
                "positive_indices",
                "positive_patients",
                "patient_disease_events",
                "outcome_admissions",
            )
        }

    def add(self, pid, index, excluded, returned, diagnoses):
        self.n0 += 1
        self.candidate_patients.add(pid)
        for j, flag in enumerate(returned):
            self.f0[j] += flag
        for disease in CODES.keys() - excluded:
            self.nd[disease] += 1
            for j, flag in enumerate(returned):
                self.fd[disease][j] += flag
            if disease in diagnoses:
                positive, outcome_hadm = diagnoses[disease]
                for j, flag in enumerate(positive):
                    self.pd[disease][j] += flag
                if positive[-1]:
                    self.at_year["positive_indices"].add(index)
                    self.at_year["positive_patients"].add(pid)
                    self.at_year["patient_disease_events"].add((pid, disease))
                    self.at_year["outcome_admissions"].add(outcome_hadm)

    def metrics(self):
        yield "all", "", "candidate_indices", self.n0
        yield "all", "", "candidate_patients", len(self.candidate_patients)
        for j, h in enumerate(HORIZONS):
            yield "all", h, "F0", self.f0[j]
        for disease in CODES:
            n = self.nd[disease]
            yield disease, "", "Nd", n
            for j, h in enumerate(HORIZONS):
                pos, ret = self.pd[disease][j], self.fd[disease][j]
                assert 0 <= pos <= ret <= min(n, self.f0[j])
                yield disease, h, "Pd", pos
                yield disease, h, "Fd", ret
                yield disease, h, "positive_rate_pct", round(100 * pos / n, 4) if n else None
        yield "all", 365, "positive_pairs", sum(v[-1] for v in self.pd.values())
        for name, items in self.at_year.items():
            yield "all", 365, name, len(items)


def followup(anchor, stays, phenotypes):
    deadlines = [anchor + timedelta(days=h) for h in HORIZONS]
    events = sorted(
        (s for s in stays if s.start > anchor and anchor < s.end <= deadlines[-1]),
        key=lambda s: (s.end, s.hadm),
    )
    returned = [bool(events) and events[0].end <= t for t in deadlines]
    diagnoses = {}
    for stay in events:
        for d in phenotypes.get(stay.hadm, ()):
            if d not in diagnoses:
                diagnoses[d] = ([stay.end <= t for t in deadlines], stay.hadm)
    return returned, diagnoses


def generate(root):
    dob, admissions, alive, phenotypes, icu_by_patient = load_data(root)
    counts = Accounting()
    for pid, history in admissions.items():
        valid = [s for s in history if s.start < s.end]
        for index in valid:
            hadm = index.hadm
            if hadm not in phenotypes or hadm not in alive:
                continue
            if (index.start - dob[pid]).days / 365.25 < 18:
                continue
            if any(s.hadm != hadm and s.start < index.end and s.end > index.start for s in history):
                continue

            icus = icu_by_patient[pid]
            first = min(
                (s for s in icus if s.hadm == hadm),
                key=lambda s: (s.start, s.end),
                default=None,
            )
            if first is None:
                continue
            prior_diagnoses = set()
            for stay in history:
                if stay.end < index.start:
                    prior_diagnoses.update(phenotypes.get(stay.hadm, ()))
            counts.add(pid, hadm, prior_diagnoses, *followup(first.end, icus, phenotypes))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="MIMIC-III v1.4 CSV directory")
    counts = generate(parser.parse_args().root)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("disease", "days", "metric", "value"))
    writer.writerows(counts.metrics())


if __name__ == "__main__":
    main()
