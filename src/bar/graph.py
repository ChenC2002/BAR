"""NetworkX-backed evidence graph preparation and incident-edge queries."""

from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path

import networkx as nx

from bar.config import Config
from bar.schemas import Edge, EvidenceGraph, Query


class GraphStore:
    def __init__(self, graph: EvidenceGraph):
        self.graph = graph
        self.nodes = {node.node_id: node for node in graph.nodes}
        self.edges = {edge.edge_id: edge for edge in graph.edges}
        self.network = nx.MultiGraph()
        self.network.add_nodes_from(self.nodes)
        for edge in graph.edges:
            self.network.add_edge(edge.head, edge.tail, key=edge.edge_id)

    @classmethod
    def load(cls, path: str | Path) -> "GraphStore":
        return cls(EvidenceGraph.model_validate_json(Path(path).read_text(encoding="utf-8")))

    def incident(self, node: str, relation: str | None = None) -> list[Edge]:
        if node not in self.network:
            return []
        ids = [key for _, _, key in self.network.edges(node, keys=True)]
        return sorted(
            (
                self.edges[eid]
                for eid in ids
                if relation is None or self.edges[eid].relation == relation
            ),
            key=lambda e: e.edge_id,
        )

    def query(self, query: Query) -> list[Edge]:
        if query.kind == "guide" and query.node_id in self.graph.expansion_guide:
            return [self.edges[eid] for eid in self.graph.expansion_guide[query.node_id]]
        return self.incident(query.node_id, query.relation)


def support_score(edge: Edge) -> float:
    """Combine precomputed source components; never discount an already complete score."""
    meta = edge.metadata
    if not any(key in meta for key in ("curated_score", "n_curated_sources", "text_score")):
        return edge.support_score
    count = float(meta.get("n_curated_sources", 0))
    curated = float(meta.get("curated_score", min(count / 3, 1)))
    text = float(meta.get("text_score", 0))
    if not isfinite(count) or not count >= 0 or not 0 <= curated <= 1 or not 0 <= text <= 1:
        raise ValueError(f"invalid source support components on {edge.edge_id}")
    return 0.4 * curated + 0.6 * text


def _anchor_path_pairs(region: nx.Graph, anchors: list[str], disease: str) -> set[frozenset]:
    # Block-cut paths give the union of simple anchor-to-disease paths without
    # enumerating exponentially many paths or retaining dangling branches.
    blocks = list(nx.biconnected_component_edges(region))
    tree = nx.Graph()
    for index, pairs in enumerate(blocks):
        for node in {node for pair in pairs for node in pair}:
            tree.add_edge(("block", index), ("node", node))
    keep = set()
    for anchor in anchors:
        if anchor == disease:
            continue
        try:
            path = nx.shortest_path(tree, ("node", anchor), ("node", disease))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            continue
        for kind, index in path:
            if kind == "block":
                keep.update(frozenset(pair) for pair in blocks[index])
    return keep


def refine_graph(
    raw: EvidenceGraph, disease: str, training_anchors: list[str], config: Config
) -> EvidenceGraph:
    """Section 2.2. Input source scores and provenance are prepared offline."""
    store = GraphStore(raw)
    if disease not in store.nodes:
        raise ValueError(f"unknown disease node: {disease}")
    region = nx.ego_graph(nx.Graph(store.network), disease, radius=config.graph.hop_limit)
    pairs = _anchor_path_pairs(region, training_anchors, disease)
    candidates = [e for e in raw.edges if frozenset((e.head, e.tail)) in pairs]
    degree = Counter(node for e in candidates for node in (e.head, e.tail))
    used = Counter()
    retained = []
    ranked = sorted(
        candidates,
        key=lambda e: (-float(e.metadata.get("prior_score", e.support_score)), e.edge_id),
    )
    for edge in ranked:
        if any(
            degree[n] > config.graph.degree_cap and used[n] >= config.graph.hub_edge_cap
            for n in (edge.head, edge.tail)
        ):
            continue
        score = support_score(edge)
        if score < config.support_threshold or not edge.provenance:
            continue
        used.update((edge.head, edge.tail))
        # Exact directed triples are the conservative default. Curated semantic
        # groups can be supplied by the caller; relation categories are not synonyms.
        group = edge.equiv_group_id or f"eq:{edge.head}|{edge.relation}|{edge.tail}"
        retained.append(edge.model_copy(update={"support_score": score, "equiv_group_id": group}))
    groups = defaultdict(list)
    guide = {}
    for edge in retained:
        groups[edge.group].append(edge.edge_id)
    for anchor in sorted(set(training_anchors) & store.nodes.keys()):
        adjacent = sorted(
            (e for e in retained if anchor in (e.head, e.tail)),
            key=lambda e: (-e.support_score, e.edge_id),
        )
        guide[anchor] = [e.edge_id for e in adjacent[: config.graph.guide_size]]
    nodes = {disease, *guide, *(n for e in retained for n in (e.head, e.tail))}
    return EvidenceGraph(
        nodes=[store.nodes[n] for n in sorted(nodes)],
        edges=retained,
        equiv_groups=dict(groups),
        expansion_guide=guide,
        metadata={"target_disease_id": disease, "support_threshold": config.support_threshold},
    )
