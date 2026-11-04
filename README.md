# Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence

**V2.0** · [Release](https://github.com/ChenC2002/BAR/tree/V2.0)

[![BAR overview: evidence graph construction, budget-aware reasoning, and training](assets/bar_overview.png)](assets/bar_overview.pdf)

BAR combines electronic health records with source-backed medical knowledge to
predict post-discharge disease risk. It assigns an uncertainty-based exploration
budget, plans and verifies graph queries, and returns a prediction with a trace
and citations. When evidence is insufficient, it returns the frozen EHR prediction.

[Demo](#demo) · [Method Map](#method-map) · [Quick Start](#quick-start) · [Data](#data) · [Workflow](#workflow)

## Demo

[![BAR Evidence Explorer](assets/bar_demo.png)](demo/index.html)

The [Evidence Explorer](demo/index.html) includes three example cases with
30-, 90-, and 365-day predictions. Change the budget, replay evidence acquisition,
inspect source records, and download a trace. The **Experiment** view includes
per-horizon results and three-task macro averages for BAR and the EHR baseline.

Predictions come from a recorded MIMIC-III CKD experiment with **30,803 patients**,
a patient-disjoint split, and a UMLS evidence graph. The viewer uses manually
defined examples and the trained checkpoint's recorded outputs with an offline
policy. It contains aggregate evaluation results and example inputs, not patient
records. These are research scores, not clinical recommendations.

The **EHR baseline** is BAR's frozen Stage-1 EHR-only predictor, following the
baseline role in Section 2.4.3. In this run it uses a 32-unit MLP encoder, the same
EHR features and task heads as BAR, and no graph evidence. Its checkpoint is
selected by validation macro average precision, then frozen for comparison,
budget allocation, and fallback.

After cloning or downloading the repository, open `demo/index.html` locally.
The viewer works offline and needs no API key, dataset, or Python installation.
GitHub displays HTML as source; use the local file to interact with it.

## Method Map

| Stage | Code | What it does |
| --- | --- | --- |
| **Evidence Graph Construction** | [`graph.py`](src/bar/graph.py) | Refine disease-specific graphs using source support, provenance, and equivalent-edge groups |
| **Budget-Aware Reasoning** | [`budget.py`](src/bar/budget.py), [`reasoning.py`](src/bar/reasoning.py), [`policy.py`](src/bar/policy.py) | Assign uncertainty-based budgets; plan, navigate, verify, revise, and stop |
| **Prediction and Training** | [`model.py`](src/bar/model.py), [`training.py`](src/bar/training.py) | Pool acquired evidence, predict risk, and train with supervised loss and paired-gain policy rewards |

Training first fits the EHR predictor and freezes its budget and fallback
reference. Uniform-routing warm-up precedes REINFORCE routing with supervised
predictor updates. Paired gains compare predictions with and without evidence
under the same current model.

## Quick Start

Use Python 3.11 or newer to run training and inference:

```bash
git clone https://github.com/ChenC2002/BAR.git
cd BAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
bar demo
```

`bar demo` opens the Evidence Explorer. Use `bar demo --no-open` to print its
location without launching a browser.

## Data

Training and prediction accept prepared JSON files following the
[input schemas](src/bar/schemas.py).

| Input | Contents |
| --- | --- |
| Evidence graph | `nodes` and `edges`, including relation, support score, provenance, and optional equivalence groups |
| Training cases | Array of `sample_id`, `disease_id`, `horizon_days`, `ehr_features`, `anchor_ids`, and binary `label` |
| Inference case | One case with the same fields, without `label` |

Disease and anchor identifiers refer to graph nodes. Feature order must stay
consistent across training and prediction. Optional `diagnosis_codes` and
`history_codes` provide context for the reasoning policy. Prepare vocabulary,
source statistics, imputation, and scaling from training data only; keep each
patient in one partition.

### MIMIC-III

The CKD runner uses ICD-9 `585*` and 30/90/365-day first-recorded diagnosis
outcomes after discharge. It combines current and historical diagnosis codes,
demographics, and pre-discharge laboratory summaries. Model selection uses
validation macro average precision, early stopping, and two preset configurations.

Required resources:

| Directory | Files |
| --- | --- |
| MIMIC-III | `ADMISSIONS.csv`, `PATIENTS.csv`, `ICUSTAYS.csv`, `DIAGNOSES_ICD.csv`, `LABEVENTS.csv` |
| Knowledge mapping | `ICD9CM_to_UMLS.csv`, `umls/umls.csv`, `umls/concept_names.txt` |

Run on a machine with authorized data access:

```bash
python -m pip install ".[experiment]"
python scripts/run_mimic_demo.py \
  --data /path/to/MIMICIII/data \
  --kg /path/to/KG_mapping \
  --out artifacts/ckd-run --patients 40000
```

The output includes the checkpoint, fitted feature transform, model-selection
record, and evaluation. Keep record-level outputs on the data host. To refresh
the viewer, export only the aggregate results and manually defined examples:

```bash
python scripts/export_demo.py artifacts/ckd-run/public_demo.json --out demo/data.js
```

## Workflow

### 1. Refine the Graph

Prepare supported source relations and choose frequent anchors from training data:

```bash
bar refine --graph data/graph.json \
  --disease C1561643 --anchor C0085580 \
  --config configs/bar_default.json --out artifacts/ckd.json
bar validate --graph artifacts/ckd.json
```

Use the disease and anchor IDs present in your graph. Refinement retains supported,
source-backed edges and prepares expansion guides for budgeted acquisition.

### 2. Train

```bash
bar train --graph artifacts/ckd.json --data data/train.json \
  --config configs/bar_default.json --policy demo --out artifacts/bar.pt
```

The [configuration](configs/bar_default.json) controls support thresholds,
budgets, evidence limits, model size, and training. `--policy demo` runs offline.
The Python `train(..., validation_cases=...)` API supports validation-based
checkpoint selection with the `experiment` extra installed.

### 3. Predict

```bash
bar predict --checkpoint artifacts/bar.pt --case data/case.json --policy demo
```

Predictions include risk, citations, and a trace of queries, verification decisions,
costs, and the stopping reason. Checkpoints retain the graph, models, and fitted
uncertainty distributions. Fallback returns the frozen EHR score and no citations.

### LLM Policy

```bash
python -m pip install ".[llm]"
export OPENAI_API_KEY=your_key
bar predict --checkpoint artifacts/bar.pt --case data/case.json \
  --policy llm --model YOUR_STRUCTURED_OUTPUT_MODEL
```

Use the same policy options with `bar train` for LLM-guided exploration.
`--base-url` selects a compatible Chat Completions endpoint supporting structured
outputs. LLM weights stay frozen; the query router is learned locally. Diagnosis
context and graph evidence are sent to the endpoint, so use only data and services
permitted by your data-use agreements. Labels and numerical EHR features are not sent.
