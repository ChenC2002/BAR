"""Run a bounded CKD experiment on its data host and export a patient-free demo."""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from bar.config import Config
from bar.graph import GraphStore, refine_graph
from bar.policy import DemoPolicy
from bar.schemas import Case, Edge, EvidenceGraph, Node, Provenance, TrainingCase
from bar.training import save_checkpoint, train

HORIZONS = (30, 90, 365)
DISEASE = "C1561643"


def rows(path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def prepare(root, max_patients, seed):
    patients = {r["SUBJECT_ID"]: r for r in rows(root / "PATIENTS.csv")}
    icu = {r["HADM_ID"] for r in rows(root / "ICUSTAYS.csv")}
    admissions = {}
    grouped = defaultdict(list)
    for r in rows(root / "ADMISSIONS.csv"):
        assert r["HADM_ID"] not in admissions
        r["start"] = datetime.fromisoformat(r["ADMITTIME"])
        r["end"] = datetime.fromisoformat(r["DISCHTIME"])
        r["codes"] = set()
        admissions[r["HADM_ID"]] = r
        grouped[r["SUBJECT_ID"]].append(r)
    for r in rows(root / "DIAGNOSES_ICD.csv"):
        if r["ICD9_CODE"]:
            assert admissions[r["HADM_ID"]]["SUBJECT_ID"] == r["SUBJECT_ID"]
            admissions[r["HADM_ID"]]["codes"].add(r["ICD9_CODE"].strip().replace(".", ""))
    eligible = defaultdict(list)
    for pid, visits in grouped.items():
        visits.sort(key=lambda v: (v["end"], v["HADM_ID"]))
        for v in visits:
            age = (v["start"] - datetime.fromisoformat(patients[pid]["DOB"])).days / 365.25
            if v["HADM_ID"] not in icu or age < 18 or v["HOSPITAL_EXPIRE_FLAG"] != "0":
                continue
            if v["end"] <= v["start"] or not v["codes"]:
                continue
            prior = [u for u in visits if u["end"] < v["start"]]
            history = set().union(*(u["codes"] for u in prior))
            if any(c.startswith("585") for c in v["codes"] | history):
                continue
            if any(u is not v and u["start"] < v["end"] and u["end"] > v["start"] for u in visits):
                continue
            event = min(
                (
                    u["end"]
                    for u in visits
                    if u["start"] > v["end"] and any(c.startswith("585") for c in u["codes"])
                ),
                default=None,
            )
            eligible[pid].append(
                {
                    "id": f"{pid}:{v['HADM_ID']}",
                    "patient": pid,
                    "time": v["end"].isoformat(),
                    "admit": v["start"].isoformat(),
                    "age": min(age, 90),
                    "male": int(patients[pid].get("GENDER", "") == "M"),
                    "stay_days": (v["end"] - v["start"]).total_seconds() / 86400,
                    "prior_visits": len(prior),
                    "codes": sorted(v["codes"]),
                    "history": sorted(history),
                    "labels": {
                        str(h): int(event is not None and event <= v["end"] + timedelta(days=h))
                        for h in HORIZONS
                    },
                }
            )
    selected = random.Random(seed).sample(sorted(eligible), min(max_patients, len(eligible)))
    selected.sort(key=lambda pid: (min(r["time"] for r in eligible[pid]), pid))
    n = len(selected)
    boundaries = (int(n * 0.7), int(n * 0.8))
    partitions = (
        selected[: boundaries[0]],
        selected[boundaries[0] : boundaries[1]],
        selected[boundaries[1] :],
    )
    assert (
        not set(partitions[0]) & set(partitions[1])
        | set(partitions[0]) & set(partitions[2])
        | set(partitions[1]) & set(partitions[2])
    )
    split = {
        name: [r for pid in ids for r in eligible[pid]]
        for name, ids in zip(("train", "validation", "test"), partitions)
    }
    summary = {
        name: {
            "patients": len(ids),
            "admissions": len(split[name]),
            "positives": {str(h): sum(r["labels"][str(h)] for r in split[name]) for h in HORIZONS},
        }
        for name, ids in zip(split, partitions)
    }
    summary["eligible_patients"] = len(eligible)
    summary["eligible_admissions"] = sum(map(len, eligible.values()))
    return split, summary


def knowledge(root, training, config):
    mapping = defaultdict(set)
    for r in rows(root / "ICD9CM_to_UMLS.csv"):
        original = r["ICD9CM"].strip().upper()
        code = original.replace(".", "")
        decimal = 4 if code.startswith("E") else 3
        canonical = code[:decimal] + ("." + code[decimal:] if len(code) > decimal else "")
        if original == canonical:
            mapping[code].add(r["UMLS"])
    # Ambiguous diagnosis-to-concept joins are not anchors.
    mapping = {code: next(iter(ids)) for code, ids in mapping.items() if len(ids) == 1}
    counts = Counter(
        cui for r in training for cui in {mapping[c] for c in r["codes"] if c in mapping}
    )
    anchors = [c for c, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:256]]
    source = root / "umls/umls.csv"
    allowed = {DISEASE, *anchors}
    with source.open() as handle:
        for relation, head, tail, _ in csv.reader(handle, delimiter="\t"):
            if DISEASE in (head, tail):
                allowed.update((head, tail))
    triples = set()
    with source.open() as handle:
        for relation, head, tail, _ in csv.reader(handle, delimiter="\t"):
            if head in allowed and tail in allowed and head != tail:
                triples.add((relation, head, tail))
    names = {}
    with (root / "umls/concept_names.txt").open() as handle:
        for line in handle:
            node, name = line.rstrip("\n").split("\t", 1)
            if node in allowed:
                names[node] = name
    edges = []
    for relation, head, tail in sorted(triples):
        key = hashlib.sha256(f"{head}|{relation}|{tail}".encode()).hexdigest()[:16]
        edges.append(
            Edge(
                edge_id=f"umls:{key}",
                head=head,
                tail=tail,
                relation=relation,
                support_score=0.4 / 3,
                metadata={"n_curated_sources": 1, "text_score": 0},
                provenance=[
                    Provenance(
                        source_id=f"UMLS:{key}",
                        title="UMLS relation export",
                        snippet=f"{names.get(head, head)} {relation} {names.get(tail, tail)}",
                    )
                ],
            )
        )
    raw = EvidenceGraph(
        nodes=[Node(node_id=n, name=names.get(n, n)) for n in sorted(allowed)], edges=edges
    )
    graph = refine_graph(raw, DISEASE, anchors, config)
    assert graph.edges
    return GraphStore(graph), mapping


def measure(labels, scores):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "ap": float(average_precision_score(labels, scores)) if sum(labels) else None,
        "brier": float(brier_score_loss(labels, scores)),
    }


def evaluate(engine, cases):
    result = {str(h): {"labels": [], "bar": [], "ehr": [], "traces": []} for h in HORIZONS}
    for sample in cases:
        prediction = engine.predict(Case.model_validate(sample.model_dump(exclude={"label"})))
        row = result[str(sample.horizon_days)]
        row["labels"].append(sample.label)
        row["bar"].append(prediction.probability)
        row["ehr"].append(prediction.trace.frozen_ehr_probability)
        row["traces"].append(prediction.trace)
        assert prediction.trace.total_cost <= prediction.trace.budget + 1e-8
    return result


def reasoning_summary(traces):
    if not traces:
        raise ValueError("Cannot summarize an empty evaluation.")
    return {
        "mean_budget": float(np.mean([t.budget for t in traces])),
        "mean_cost": float(np.mean([t.total_cost for t in traces])),
        "fallback_rate": float(np.mean([t.fallback for t in traces])),
        "budget_violations": sum(t.total_cost > t.budget + 1e-8 for t in traces),
    }


def main():
    import joblib
    from mimic_features import FeatureEncoder, add_laboratories

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    split, cohort = prepare(args.data, args.patients, args.seed)
    print(json.dumps({"stage": "cohort", **cohort}), flush=True)
    add_laboratories(args.data, split)
    encoder = FeatureEncoder().fit(split["train"])
    feature_rows = {
        r["id"]: features.tolist()
        for records in split.values()
        for r, features in zip(records, encoder.transform(records), strict=True)
    }
    joblib.dump(encoder, args.out / "features.joblib")
    config = Config(
        seed=args.seed,
        hidden_dim=32,
        support_threshold=0.1,
        graph={"hop_limit": 2, "degree_cap": 50, "hub_edge_cap": 20},
        reasoning={"min_quality": 0.03, "fallback_quality": 0.03},
        training={
            "baseline_epochs": 40,
            "warmup_epochs": 2,
            "policy_epochs": 3,
            "batch_size": 128,
            "episodes_per_round": 1024,
            "predictor_lr": 0.0003,
            "finetune_lr": 0.00003,
            "weight_decay": 0.05,
            "patience": 6,
        },
    )
    store, mapping = knowledge(args.kg, split["train"], config)

    def convert(row, horizon, labeled=True):
        features = feature_rows[row["id"]] if labeled else encoder.transform([row])[0].tolist()
        anchors = sorted(
            {
                mapping[c]
                for c in row["codes"]
                if c in mapping and mapping[c] in store.nodes and store.incident(mapping[c])
            }
        )[:12]
        fields = dict(
            sample_id=row["id"],
            disease_id=DISEASE,
            horizon_days=horizon,
            ehr_features=features,
            anchor_ids=anchors,
            diagnosis_codes=row["codes"],
            history_codes=row["history"],
        )
        return (
            TrainingCase(**fields, label=row["labels"][str(horizon)]) if labeled else Case(**fields)
        )

    prepared = {
        name: [convert(r, h) for r in records for h in HORIZONS] for name, records in split.items()
    }
    (args.out / "cohort.json").write_text(json.dumps(split))
    (args.out / "feature_layout.json").write_text(json.dumps(encoder.vocabulary))
    (args.out / "config.json").write_text(config.model_dump_json(indent=2))
    print(
        json.dumps(
            {
                "stage": "training",
                "cases": len(prepared["train"]),
                "nodes": len(store.nodes),
                "edges": len(store.edges),
            }
        ),
        flush=True,
    )
    candidates = []
    for hidden, lr, decay in [(32, 0.0003, 0.05), (64, 0.001, 0.01)]:
        candidate = config.model_copy(
            update={
                "hidden_dim": hidden,
                "training": config.training.model_copy(
                    update={"predictor_lr": lr, "weight_decay": decay}
                ),
            }
        )
        engine, training = train(
            prepared["train"],
            {DISEASE: store},
            DemoPolicy(),
            candidate,
            validation_cases=prepared["validation"],
        )
        score = max(row["validation_ap"] for row in training if row["stage"] != "ehr_only")
        candidates.append((score, engine, training))
        print(
            json.dumps(
                {
                    "stage": "candidate",
                    "hidden": hidden,
                    "validation_ap": score,
                    "epochs": len(training),
                }
            ),
            flush=True,
        )
    _, engine, training = max(candidates, key=lambda item: item[0])
    config = engine.config
    (args.out / "config.json").write_text(config.model_dump_json(indent=2))
    save_checkpoint(engine, args.out / "bar.pt")
    print(json.dumps({"stage": "validation"}), flush=True)
    validation = evaluate(engine, prepared["validation"])
    # Validation-only choice between two completed stages, fixed before test evaluation.
    val_ap = {}
    for model in ("bar", "ehr"):
        available = [measure(v["labels"], v[model])["ap"] for v in validation.values()]
        available = [value for value in available if value is not None]
        val_ap[model] = float(np.mean(available)) if available else None
    if val_ap["bar"] is None:
        raise ValueError(
            "No validation positives; increase --patients for a measurable experiment."
        )
    preferred = max(val_ap, key=val_ap.get)
    print(
        json.dumps({"stage": "test", "validation_ap": val_ap, "validation_preferred": preferred}),
        flush=True,
    )
    (args.out / "selection.json").write_text(
        json.dumps(
            {
                "metric": "validation macro average precision",
                "candidates": [
                    {"hidden": e.config.hidden_dim, "validation_ap": s, "history": hist}
                    for s, e, hist in candidates
                ],
            }
        )
    )
    testing = evaluate(engine, prepared["test"])
    metrics = []
    for h in HORIZONS:
        test_h = [c for c in prepared["test"] if c.horizon_days == h]
        result = testing[str(h)]
        metrics.append(
            {
                "horizon": h,
                "samples": len(test_h),
                "positives": sum(result["labels"]),
                "prevalence": float(np.mean(result["labels"])),
                "ehr": measure(result["labels"], result["ehr"]),
                "bar": measure(result["labels"], result["bar"]),
            }
        )
    traces = [t for row in testing.values() for t in row["traces"]]
    summary = {
        "dataset": "MIMIC-III",
        "task": "First recorded CKD diagnosis",
        "seed": args.seed,
        "cohort": cohort,
        "metrics": metrics,
        "validation_ap": val_ap,
        "validation_preferred": preferred,
        "reasoning": reasoning_summary(traces),
        "graph": {
            "source": "Local UMLS export",
            "nodes": len(store.nodes),
            "edges": len(store.edges),
            "support_threshold": 0.1,
        },
        "policy": "Offline deterministic policy",
        "encoder": "ICD codes, demographics, pre-discharge labs + MLP",
        "baseline": {
            "name": f"BAR Stage-1 EHR-only MLP ({config.hidden_dim} hidden units)",
            "hidden_dim": config.hidden_dim,
            "selection": "Best Stage-1 validation macro AP within the selected BAR configuration",
            "frozen": True,
        },
        "selection": "Validation macro AP; two preset MLP configurations, early stopping",
        "training_patients": cohort["train"]["patients"],
        "split": "70/10/20 patient groups ordered by earliest eligible discharge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "training.json").write_text(json.dumps(training))
    (args.out / "test_predictions.json").write_text(
        json.dumps(
            {h: {k: v for k, v in row.items() if k != "traces"} for h, row in testing.items()}
        )
    )
    examples = [
        {
            "id": "example-diabetes",
            "name": "Diabetes + hypertension",
            "codes": ["25000", "4019"],
            "history": ["25000"],
            "age": 65,
            "male": 1,
            "stay_days": 5,
            "prior_visits": 1,
            "labs": {
                "creatinine": [1.6, 1.3, 2.1],
                "urea_nitrogen": [32, 25, 44],
                "potassium": [4.6, 4.2, 5.0],
                "hemoglobin": [10.5, 10.0, 11.2],
            },
            "labels": ["Type 2 diabetes", "Hypertension"],
        },
        {
            "id": "example-heart",
            "name": "Heart failure + hypertension",
            "codes": ["4280", "4019"],
            "history": ["4019"],
            "age": 72,
            "male": 1,
            "stay_days": 7,
            "prior_visits": 1,
            "labs": {
                "creatinine": [1.3, 1.1, 1.8],
                "urea_nitrogen": [28, 20, 36],
                "potassium": [4.2, 3.6, 4.8],
                "hemoglobin": [11.5, 11.0, 12.2],
            },
            "labels": ["Heart failure", "Hypertension"],
        },
        {
            "id": "example-infection",
            "name": "Pneumonia",
            "codes": ["486"],
            "history": [],
            "age": 45,
            "male": 0,
            "stay_days": 4,
            "prior_visits": 0,
            "labs": {
                "creatinine": [0.8, 0.7, 0.9],
                "urea_nitrogen": [14, 12, 18],
                "potassium": [4.0, 3.8, 4.4],
                "hemoglobin": [13.0, 12.5, 13.5],
            },
            "labels": ["Pneumonia"],
        },
    ]
    public_cases = []
    for example in examples:
        outputs = {}
        for h in HORIZONS:
            case = convert(example, h, labeled=False)
            for budget in ("auto", "2", "6", "12"):
                original = engine.config
                if budget != "auto":
                    engine.config = original.model_copy(
                        update={
                            "reasoning": original.reasoning.model_copy(
                                update={"min_budget": float(budget), "max_budget": float(budget)}
                            )
                        }
                    )
                prediction = engine.predict(case)
                engine.config = original
                ids = {
                    e for step in prediction.trace.steps for e in step.candidates + step.selected
                } | set(prediction.trace.citation_ids)
                edges = [store.edges[eid] for eid in sorted(ids)]
                node_ids = {
                    DISEASE,
                    *case.anchor_ids,
                    *(node for e in edges for node in (e.head, e.tail)),
                }
                outputs[f"{h}:{budget}"] = {
                    **prediction.model_dump(),
                    "anchors": case.anchor_ids,
                    "nodes": [store.nodes[n].model_dump() for n in sorted(node_ids)],
                    "edges": [e.model_dump() for e in edges],
                }
        public_cases.append(
            {
                "id": example["id"],
                "name": example["name"],
                "diagnoses": example["labels"],
                "history": example["history"],
                "inputs": {
                    key: example[key]
                    for key in (
                        "codes",
                        "history",
                        "age",
                        "male",
                        "stay_days",
                        "prior_visits",
                        "labs",
                    )
                },
                "clinical": {
                    "age": example["age"],
                    "sex": "Male" if example["male"] else "Female",
                    "creatinine": example["labs"]["creatinine"][0],
                },
                "outputs": outputs,
            }
        )
    public = {
        "summary": summary,
        "cases": public_cases,
        "target": DISEASE,
        "case_origin": "Manually defined examples evaluated with the trained checkpoint",
    }
    (args.out / "public_demo.json").write_text(json.dumps(public, indent=2))
    print(json.dumps({"stage": "complete", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
