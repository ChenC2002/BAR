"""Cohort construction for first-onset post-discharge prediction."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from bar.schemas import CohortSample, PatientVisit


def parse_time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        timestamp.replace(tzinfo=timezone.utc)
        if timestamp.tzinfo is None
        else timestamp.astimezone(timezone.utc)
    )


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

                anchors = [
                    code_to_node[code] for code in visit.diagnosis_codes if code in code_to_node
                ]
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
    return any(
        parse_time(visit.admit_time) > discharge_time
        and discharge_time
        < parse_time(visit.discharge_time)
        <= discharge_time + timedelta(days=window_days)
        and bool(target_codes & set(visit.diagnosis_codes))
        for visit in future_visits
    )


def temporal_split_samples(
    samples: Sequence[CohortSample],
    train_fraction: float = 0.7,
    valid_fraction: float = 0.1,
) -> Tuple[List[CohortSample], List[CohortSample], List[CohortSample]]:
    """Order patients by earliest eligible discharge; keep every patient's rows together.

    Fractions apply to patients. This explicit grouping convention prevents patient
    overlap, but does not guarantee non-overlapping calendar ranges across splits.
    """

    if not (
        0 <= train_fraction <= 1
        and 0 <= valid_fraction <= 1
        and train_fraction + valid_fraction <= 1
    ):
        raise ValueError(
            "train_fraction and valid_fraction must be non-negative and sum to at most 1"
        )

    grouped: Dict[str, List[CohortSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.patient_id].append(sample)

    for group in grouped.values():
        group.sort(key=lambda sample: (parse_time(sample.index_time), sample.sample_id))

    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: (
            parse_time(group[0].index_time),
            group[0].patient_id,
            group[0].index_visit_id,
        ),
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
