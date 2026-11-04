# BAR

Core implementation of **Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence**.

BAR builds disease-specific evidence graphs, acquires evidence through a budget-aware plan-navigate-verify loop, and combines that evidence with EHR features for risk prediction. This release implements the core algorithm in Python and PyTorch, with Pydantic input validation and NetworkX graph operations.

## Quick Start

Python 3.11 or newer. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
bar demo --config configs/bar_default.json --out artifacts/demo.pt
pytest -q
```

The demo trains all three stages on eight synthetic cases, then predicts a separate synthetic case. It runs on CPU without an API key. Its offline policy exercises graph navigation; it does not perform clinical semantic verification. The printed risk score is a software example, not a paper result.

## Core Algorithm

| Component | Implementation |
| --- | --- |
| Evidence graph | `graph.py`: K-hop region, anchor-to-disease paths, hub control, support/provenance filtering, equivalence groups and expansion guides |
| Patient budget | `budget.py`: task-specific training uncertainty CDF, fixed for later inference; default budget range 2-12 |
| Patient state and prediction | `model.py`: target embedding, GRU state updates, support-times-relevance quality, weighted edge pooling and task-specific risk heads |
| Reasoning | `reasoning.py`: planning, admissible query routing, selection, verification, plan replacement, stopping and frozen-baseline fallback |
| Training | `training.py`: EHR baseline, uniform-routing warm-up, then REINFORCE routing plus supervised predictor updates |
| LLM calls | `policy.py`: typed plan/select/verify/revise outputs through the official Python SDK |

The first query uses an anchor. Later queries expand only anchors or endpoints of accepted evidence. Every executed query and selection is charged, including empty selections and failed verification. Evidence and patient state change only on verified steps. Citations retain source records and use the final patient-state scores.

## Prepared Inputs

See [the graph](examples/tiny_evidence_graph.json), [training cases](examples/tiny_cases.json), and [an inference case](examples/tiny_case.json). `ehr_features` is a fixed-length vector prepared from discharge-time EHR data; the compact reference encoder is an MLP. Use the same feature layout at training and inference. Labels are only present in training inputs and are never sent to the LLM.

Graph refinement accepts source scores prepared offline: `metadata.n_curated_sources` and `metadata.text_score` produce `0.4 * min(count / 3, 1) + 0.6 * text_score`. If no component scores are supplied, the existing `support_score` is treated as complete. Provenance is required for retained edges. Supply curated `equiv_group_id` values for semantic deduplication; otherwise only identical directed triples are grouped.

```bash
bar refine --graph examples/tiny_evidence_graph.json --disease CKD \
  --anchor DIABETES --anchor HTN --out artifacts/ckd.json
bar train --graph artifacts/ckd.json --data examples/tiny_cases.json \
  --policy demo --out artifacts/bar.pt
bar predict --checkpoint artifacts/bar.pt --case examples/tiny_case.json --policy demo
```

`train` consumes training cases only. It fits uncertainty CDFs after freezing the baseline. Checkpoints include model weights, the frozen baseline, query router, vocabularies, configuration, CDFs, and graph snapshots. They are inference checkpoints, not resumable optimizer checkpoints. The CLI handles one disease graph per run; the Python `train` API accepts a dictionary of disease graphs and multiple disease/horizon tasks.

## LLM Policy

```bash
pip install ".[llm]"
export OPENAI_API_KEY=your_key
bar predict --checkpoint artifacts/bar.pt --case examples/tiny_case.json \
  --policy llm --model YOUR_STRUCTURED_OUTPUT_MODEL
```

The same `--policy llm --model ...` options work for training. A compatible endpoint can be provided with `--base-url`; it must support Chat Completions structured outputs. The SDK handles transport retries and timeouts. Refusals and malformed outputs produce an explicit `policy_error` trace and frozen-EHR fallback. LLM weights stay frozen; REINFORCE trains the local query router. The demo-trained checkpoint is only suitable for checking integration.

## Scope

This is a compact algorithm implementation of Sections 2.2-2.4, not the full experiment pipeline. It intentionally omits MIMIC-specific ingestion, PubMed retrieval/embedding generation, large-scale training infrastructure, baseline suites, citation-judge studies and appendix experiments. The training schedule and model width in [the config](configs/bar_default.json) are small example settings, not the paper's experimental configuration.

The optional `data/` helpers label outcomes by subsequent discharge and keep each patient's admissions in one split, ordering patient groups by earliest eligible discharge. That grouping convention does not imply strictly disjoint calendar ranges. Clinical eligibility checks and feature extraction belong in the upstream data preparation.

For the implementation choices and alternatives, see [Engineering Decisions](docs/engineering.md).
