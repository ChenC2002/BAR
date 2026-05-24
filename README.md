# Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence

The project studies post-discharge risk prediction from electronic health records (EHRs). Given a patient's discharge-time diagnosis codes, previous diagnosis history, a target disease, and a future prediction horizon, BAR predicts whether the disease will first appear after discharge and returns a compact citation list of supporting medical-knowledge-graph edges.

The implementation follows the paper's terms directly:

1. **Refined evidence graphs**: convert a raw biomedical KG into disease-specific evidence graphs `G_d` whose edges are supported, traceable, and low-redundancy.
2. **Budget-aware plan-navigate-verify reasoning**: allocate a patient-specific evidence-acquisition budget, query the graph, verify selected evidence, and stop when useful evidence is exhausted.
3. **Evidence-aware prediction and training**: combine EHR signal and cited KG evidence, then optimize paired prediction gain, acquisition cost, and citation integrity.

The repository currently includes the core data contracts, graph-refinement logic, reasoning loop, reward definitions, evaluation helpers, script entry points, and a tiny runnable example. Full paper-scale experiments require credentialed clinical data and optional ML/LLM dependencies.

## Quick Start

The quick start uses a tiny synthetic graph, so it does not require MIMIC, PrimeKG, PubMed, GPUs, or an LLM API key.

Run the smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/smoke_test.py
```

Validate the example evidence graph:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m bar.cli validate-evidence-graph \
  --graph examples/tiny_evidence_graph.json
```

Run one budget-aware reasoning trace:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m bar.cli demo-reasoning \
  --graph examples/tiny_evidence_graph.json \
  --anchor DIABETES \
  --disease CKD \
  --ehr-score 0.5
```

Show the configured training stages:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m bar.cli show-training-stages
```

Expected demo output includes a patient-specific budget, total acquisition cost, selected evidence edges, cited edges, fallback status, and risk probability.

## Installation

Minimal demos and tests use only the Python standard library:

```bash
python3 --version
PYTHONPATH=src python3 -m bar.cli show-training-stages
```

For full experiments, install optional dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

or:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
src/bar/
  data/
    cohort.py           First-onset post-discharge cohort construction
    linker.py           ICD -> UMLS CUI -> PrimeKG anchor linker
  evidence/
    refinement.py       Disease-specific evidence graph construction
  kg/
    graph_store.py      Evidence graph loading, validation, and traversal
  reasoning/
    loop.py             Budget-aware plan-navigate-verify loop
    query_engine.py     QUERY_GUIDE and QUERY_EXPAND execution
    scoring.py          Patient-conditioned evidence-quality scoring
    prompts.py          LLM prompts for plan/select/verify/revise/citation judge
  prediction/
    predictor.py        Predictor interface and citation formation
  training/
    rewards.py          Paired gain, cost, and citation-integrity reward
    reinforce.py        REINFORCE helper with running-mean baseline
    stages.py           Three-stage training procedure
  eval/
    metrics.py          AUROC, average precision, PR curve
    citation.py         Citation precision judge utilities

configs/bar_default.json
examples/tiny_evidence_graph.json
scripts/
tests/smoke_test.py
```

## Method And Code Map

| Method component | Code path | What it does |
| --- | --- | --- |
| First-onset cohort | `bar.data.cohort` | Builds samples where the target disease has not appeared before the index discharge and labels future onset within 30/90/365 days. |
| Temporal split | `bar.data.cohort.temporal_split_samples` | Splits patient/index-admission groups chronologically so target/horizon rows from the same admission stay in the same split. |
| ICD-to-KG linker | `bar.data.linker` | Maps ICD codes to UMLS CUIs, then to PrimeKG node ids. The linker is fixed and not updated during training. |
| Refined evidence graph `G_d` | `bar.evidence.refinement` | Selects K-hop disease regions, keeps anchor-reachable paths, applies hub control, scores support, groups equivalent edges, and builds expansion guide `I_d`. |
| Support score `s_e` | `SupportScorer` | Combines curated-source support and textual evidence score when both are available; preserves an existing prior score when no component metadata is present. |
| `QUERY_GUIDE` / `QUERY_EXPAND` | `bar.reasoning.query_engine` | Executes the two graph-query types used by the reasoning loop. |
| Evidence-quality score `rho_t(e)` | `bar.reasoning.scoring` | Scores an edge by both support and patient relevance. The current implementation is dependency-free; paper-scale runs should plug in learned patient-state and node embeddings. |
| Plan-navigate-verify loop | `bar.reasoning.loop` | Applies patient-specific budget, candidate prefiltering, admissibility checks, verification, revision hook, stopping, fallback, and citation selection. |
| LLM prompts | `bar.reasoning.prompts` | Plan generation, edge selection, verification, plan revision, and citation judge templates. |
| Predictor | `bar.prediction.predictor` | Consumes EHR signal and score-weighted evidence; returns risk probability, fallback status, and cited edge ids. |
| Reward | `bar.training.rewards` | Implements paired gain `L_raw - L_aug`, normalized acquisition cost, and citation integrity. |
| Policy optimization | `bar.training.reinforce` | Provides the REINFORCE loss helper used in Stage 3. |
| Metrics | `bar.eval.metrics`, `bar.eval.citation` | Prediction quality, citation precision, and reasoning-quality utilities. |

## Evidence Graph Format

The refined evidence graph is the key artifact passed from graph construction to reasoning:

```json
{
  "nodes": [
    {
      "node_id": "CKD",
      "name": "Chronic kidney disease",
      "node_type": "target_disease"
    }
  ],
  "edges": [
    {
      "edge_id": "e1",
      "head": "DIABETES",
      "relation": "risk_factor_for",
      "tail": "CKD",
      "support_score": 0.91,
      "equiv_group_id": "eq:DIABETES|risk_factor_for|CKD",
      "provenance": [
        {
          "source_id": "pmid:...",
          "title": "source title",
          "score": 0.9
        }
      ],
      "metadata": {
        "relation_category": "disease_disease",
        "n_curated_sources": 2,
        "text_score": 0.88
      }
    }
  ],
  "equiv_groups": {
    "eq:DIABETES|risk_factor_for|CKD": ["e1"]
  },
  "expansion_guide": {
    "DIABETES": ["e1"]
  }
}
```

Important fields:

- `support_score`: edge-level evidential support `s_e`.
- `provenance`: source identifiers or retrieved snippets used to support the edge.
- `equiv_group_id`: groups semantically equivalent edges so citation lists avoid repetition.
- `expansion_guide`: precomputed `I_d` entries for fast `QUERY_GUIDE(anchor)` calls.

## Reasoning Loop

For each patient `i`, target disease `d`, and horizon `h`, BAR runs:

1. Link discharge-time codes to KG anchors `A_i,d`.
2. Allocate patient-specific budget `B_i` from EHR-only uncertainty.
3. Ask the policy for a pathway-oriented plan.
4. Use `QUERY_GUIDE(anchor)` for the first step and `QUERY_EXPAND(concept, relation)` for later steps.
5. Score candidates by support and patient relevance.
6. Keep top candidates, remove already-seen equivalence groups, and ask the policy to select evidence.
7. Enforce hard admissibility checks: edge exists, support is high enough, group is not repeated.
8. Verify whether the selected evidence supports the plan-step hypothesis.
9. Revise failed plan steps when evidence is absent, contradictory, or weak.
10. Stop when budget is exhausted, evidence utility stagnates, or the evidence cap is reached.
11. Fall back to EHR-only prediction when the evidence set is too small or too weak.
12. Return risk score and cited edge ids.

The runnable demo uses `HeuristicReasoningPolicy` so the loop can be tested locally. Full experiments should replace that policy with an LLM-backed implementation using `bar.reasoning.prompts`.

## Training Procedure

BAR follows three stages:

1. **Stage 1: EHR-only baseline**
   - Set `z_kg = 0`.
   - Train the EHR encoder and predictor with binary cross-entropy.
   - Save `y_ehr` and `L_raw`; freeze this baseline for later stages.

2. **Stage 2: supervised warm-up**
   - Run the reasoning loop with a fixed evidence-acquisition policy.
   - Train the predictor while conditioning on acquired evidence.
   - This stage checks that the predictor can use cited evidence before policy optimization.

3. **Stage 3: REINFORCE policy optimization**
   - Optimize the reasoning policy using episode reward:
     - paired gain: `L_raw - L_aug`
     - normalized acquisition cost: `C_i,d,h / B_max`
     - citation integrity: cited edges exist, meet support threshold, avoid repeated equivalence groups, fit citation cap, and have provenance
   - Continue supervised predictor updates.

Inspect the configured stages:

```bash
PYTHONPATH=src python3 -m bar.cli show-training-stages
```

## Script Entry Points

Restricted clinical data are not redistributed. The scripts in `scripts/` define the intended full-run boundaries and validate expected inputs, but they are intentionally scaffold entry points until local dataset loaders, PrimeKG/PubMed ingestion, neural models, and LLM clients are connected.

Each script writes a small manifest or schema file so users can confirm paths and hand-off contracts before filling the paper-scale implementation:

```bash
# 1. Prepare temporal first-onset cohort artifacts.
PYTHONPATH=src python3 scripts/prepare_cohorts.py \
  --config configs/bar_default.json \
  --mimic-root /path/to/mimic \
  --out data/processed

# 2. Prepare one refined evidence-graph artifact directory per target disease.
PYTHONPATH=src python3 scripts/refine_evidence_graphs.py \
  --config configs/bar_default.json \
  --primekg /path/to/primekg \
  --pubmed-index /path/to/pubmed \
  --cohorts data/processed \
  --out artifacts/evidence_graphs

# 3. Reserve the Stage 1 EHR-only baseline output contract.
PYTHONPATH=src python3 scripts/train_ehr_baseline.py \
  --config configs/bar_default.json \
  --cohorts data/processed \
  --out artifacts/stage1

# 4. Reserve the Stage 2 supervised warm-up output contract.
PYTHONPATH=src python3 scripts/train_warmup.py \
  --config configs/bar_default.json \
  --graphs artifacts/evidence_graphs \
  --stage1 artifacts/stage1 \
  --out artifacts/stage2

# 5. Reserve the Stage 3 REINFORCE output contract.
PYTHONPATH=src python3 scripts/train_reinforce.py \
  --config configs/bar_default.json \
  --stage2 artifacts/stage2 \
  --out artifacts/stage3

# 6. Write the evaluation metric schema.
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/bar_default.json \
  --checkpoint artifacts/stage3 \
  --out results
```

## Configuration

Default settings live in [configs/bar_default.json](configs/bar_default.json):

- `evidence_graph`: hop limit, support threshold, hub control, support-scorer weights, expansion-guide size
- `reasoning_loop`: budget bounds, query/expand/select costs, prefilter cap, evidence cap, citation cap, stop/fallback thresholds
- `llm`: default and secondary model names for future LLM-backed policy implementations
- `training`: stage names, optimizer choice, learning rates, encoder/fusion descriptions, reward weights
- `experiments`: datasets, horizons, target diseases, and reported metrics

The `reasoning_loop` block is intentionally compatible with `ReasoningBudget(**config["reasoning_loop"])`.

## Data And Assets

Full experiments expect:

- MIMIC-III and/or MIMIC-IV diagnosis/admission tables under credentialed access.
- ICD-to-UMLS mappings and CUI-to-PrimeKG mappings.
- PrimeKG nodes and edges.
- A fixed PubMed abstract index for textual support scoring.
- Optional LLM access for plan generation, edge selection, verification, plan revision, and citation judging.

The repository does not include restricted clinical data.

## Development Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/smoke_test.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 - <<'PY'
modules = [
  "bar.data.cohort", "bar.data.linker", "bar.evidence.refinement",
  "bar.kg.graph_store", "bar.reasoning.loop", "bar.prediction.predictor",
  "bar.training.rewards", "bar.eval.metrics"
]
for module in modules:
    __import__(module)
print("imports ok")
PY
```

## Current Scope

Implemented now:

- Stable schemas and JSON evidence graph contract
- Disease-specific evidence graph refinement logic
- `QUERY_GUIDE` / `QUERY_EXPAND` graph access
- Budget-aware plan-navigate-verify loop
- Prompt templates
- Predictor interface
- Reward and metric utilities
- Tiny local example and smoke test
- Scaffold script entry points for full-run boundaries

Still to fill for paper-scale runs:

- Concrete MIMIC table loaders
- PrimeKG/PubMed ingestion
- PubMed BM25/PubMedBERT support-score computation
- Neural EHR encoder and edge representation model
- LLM policy implementation
- End-to-end training loops behind the script entry points
