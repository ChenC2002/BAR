# Engineering Decisions

This implementation follows the core method in the supplied October 2026 manuscript. The following comparison is based on public engineering documentation, not on claims about any company's private infrastructure or a market-share survey.

## Three Options

| Option | Appropriate use | Trade-off |
| --- | --- | --- |
| **Explicit Python loop + PyTorch + Pydantic + NetworkX (chosen)** | A small, inspectable algorithm implementation and offline experiments | Easy to test and change; execution and graph storage stay within one process |
| LangGraph + the same numerical modules | Long-running inference requiring persistent state, recovery, streaming or human intervention | Reuses workflow infrastructure, but adds state serialization and orchestration dependencies |
| Neo4j + a model service and workflow runtime | Shared graph access across many clients and larger operational deployments | Centralized graph querying and administration, at the cost of database operations and service boundaries |

[Anthropic's agent engineering guidance](https://www.anthropic.com/research/building-effective-agents) recommends starting with simple composable patterns and direct APIs. It is architectural guidance rather than evidence of one universal industry stack. [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) provides durable execution, streaming and human intervention. [Neo4j GDS](https://neo4j.com/docs/graph-data-science/current/algorithms/) supplies graph algorithms and an established database ecosystem. Our choice is an inference from these capabilities and this repository's deliberately small scope.

## What To Reuse

| Previous issue | Current approach |
| --- | --- |
| Handwritten schema parsing and silent config drift | [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) validates configuration, graph references and LLM responses; unknown fields fail early |
| Handwritten traversal and overly broad reachability | [NetworkX](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html) handles graph structure; block-cut paths identify anchor-to-disease paths without exponential path enumeration |
| Hashed node relevance and a scalar heuristic risk score | PyTorch embeddings, GRUCell, differentiable quality/pooling and task heads |
| A float-only "REINFORCE loss" with no learning | [Categorical distributions and log probabilities](https://docs.pytorch.org/docs/stable/distributions.html), autograd and separate AdamW optimizers |
| Handwritten BCE and ranking metrics | [PyTorch BCE with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html); remove unused experiment metrics rather than carry another metrics library |
| Prompt strings without a real integration boundary | [Official SDK parsing](https://github.com/openai/openai-python/blob/main/helpers.md), typed outputs, SDK retries and timeouts |
| Several scripts that only write manifests | One CLI whose commands actually refine, train, save or predict |

These are the issues an infrastructure-oriented review should focus on: correctness at boundaries, unnecessary custom infrastructure, and whether commands do what they promise. More layers, agents, factories, or a graph database would not solve those issues by themselves.

## Algorithm Invariants

- Budgets use a per-task empirical training CDF, including right-inclusive ties; no fitting on validation/test predictions.
- GUIDE and EXPAND are mutually exclusive. Selection is charged even when empty or rejected. Retry transport does not duplicate acquisition charges.
- The LLM can only select presented candidates. Group uniqueness, support, quality and remaining evidence capacity are enforced before semantic verification.
- Failed verification preserves evidence and state; revision replaces the entire unexecuted plan. Successful verification updates the GRU state and rescoring uses that new state.
- Fallback returns the frozen stage-1 probability with no citations. Paired gain instead uses the current predictor with zero/nonzero evidence at the same parameter snapshot and with dropout disabled.
- Routing gradients affect only the router. Supervised gradients update the encoder, embeddings, state transition and predictor. The reward is detached, failed actions retain their log probabilities, and the moving reward baseline updates after the policy step.
- Inference checkpoints carry the input vocabularies, graph snapshots and fitted CDFs with the weights.

## Deliberate Simplifications

EHR inputs are precomputed feature vectors with a small MLP, not the appendix's full longitudinal encoder. Source retrieval and semantic equivalence annotation are external preprocessing. The router uses a small configurable hidden width. Training uses in-memory full batches; there is no distributed runner, serving layer, paper-scale benchmark suite or experiment tracker. Add standard tooling for these only when the corresponding workload exists.

The LLM adapter uses Chat Completions for compatible endpoint portability. It is not a universal provider adapter; an endpoint must support the SDK's structured-output contract. Live LLM evaluation is separate from the offline deterministic tests.
