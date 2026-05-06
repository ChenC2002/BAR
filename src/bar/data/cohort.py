"""Cohort construction for first-onset post-discharge prediction."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from bar.schemas import CohortSample, PatientVisit


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_first_onset_samples(
    visits: Iterable[PatientVisit],
    target_code_sets: Dict[str, Set[str]],
    windows_days: Sequence[int],
    code_to_node: Dict[str, str],
) -> List[CohortSample]:
    """Create post-discharge first-onset samples.

    A visit is excluded for a target disease if the target code appears in the
    current visit or any prior visit. Labels are assigned from future visits
    within the requested window after discharge.
    """

    by_patient: Dict[str, List[PatientVisit]] = defaultdict(list)
    for visit in visits:
        by_patient[visit.patient_id].append(visit)
    for patient_visits in by_patient.values():
        patient_visits.sort(key=lambda visit: parse_time(visit.discharge_time))

    samples: List[CohortSample] = []
    for patient_id, patient_visits in by_patient.items():
        history_codes: List[str] = []
        for index, visit in enumerate(patient_visits):
            current_codes = set(visit.diagnosis_codes)
            future = patient_visits[index + 1 :]
            discharge_time = parse_time(visit.discharge_time)

            for target_id, target_codes in target_code_sets.items():
                if target_codes & set(history_codes):
                    continue
                if target_codes & current_codes:
                    continue

                anchors = [code_to_node[code] for code in visit.diagnosis_codes if code in code_to_node]
                for window_days in windows_days:
                    label = _has_future_onset(future, target_codes, discharge_time, window_days)
                    sample_id = f"{patient_id}:{visit.visit_id}:{target_id}:{window_days}"
                    samples.append(
                        CohortSample(
                            sample_id=sample_id,
                            patient_id=patient_id,
                            index_visit_id=visit.visit_id,
                            index_time=visit.discharge_time,
                            target_disease_id=target_id,
                            window_days=int(window_days),
                            label=int(label),
                            diagnosis_codes=list(visit.diagnosis_codes),
                            history_codes=list(history_codes),
                            anchor_node_ids=anchors,
                        )
                    )
            history_codes.extend(visit.diagnosis_codes)
    return samples


def _has_future_onset(
    future_visits: Sequence[PatientVisit],
    target_codes: Set[str],
    discharge_time: datetime,
    window_days: int,
) -> bool:
    for visit in future_visits:
        delta_days = (parse_time(visit.admit_time) - discharge_time).days
        if delta_days < 0:
            continue
        if delta_days > window_days:
            return False
        if target_codes & set(visit.diagnosis_codes):
            return True
    return False


def temporal_split_samples(
    samples: Sequence[CohortSample],
    train_fraction: float = 0.7,
    valid_fraction: float = 0.1,
) -> Tuple[List[CohortSample], List[CohortSample], List[CohortSample]]:
    """Chronologically split patient/index-admission groups.

    Cohort construction expands one index admission into multiple target-disease
    and horizon rows. Splitting after expansion can place rows from the same
    admission in different splits, so this function groups by patient, index
    visit, and index time before applying temporal boundaries.
    """

    if train_fraction < 0.0 or valid_fraction < 0.0 or train_fraction + valid_fraction > 1.0:
        raise ValueError("train_fraction and valid_fraction must be non-negative and sum to at most 1")

    grouped: Dict[Tuple[str, str, str], List[CohortSample]] = defaultdict(list)
    for sample in samples:
        key = (sample.patient_id, sample.index_visit_id, sample.index_time)
        grouped[key].append(sample)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: (parse_time(group[0].index_time), group[0].patient_id, group[0].index_visit_id),
    )
    n_groups = len(ordered_groups)
    train_end = int(n_groups * train_fraction)
    valid_end = train_end + int(n_groups * valid_fraction)

    def flatten(groups: Sequence[List[CohortSample]]) -> List[CohortSample]:
        return [sample for group in groups for sample in group]

    return (
        flatten(ordered_groups[:train_end]),
        flatten(ordered_groups[train_end:valid_end]),
        flatten(ordered_groups[valid_end:]),
    )
