"""Disease-specific refined evidence graph construction.

This implements the paper's Stage 1 terminology directly:

1. collect a K-hop candidate region around a target disease,
2. restrict it to paths reachable from frequent discharge-time anchors,
3. suppress high-degree hubs,
4. compute support scores from curated and textual evidence,
5. assign equivalence-group ids, and
6. build the expansion guide I_d used by QUERY_GUIDE.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from bar.schemas import Edge, EvidenceGraph, Node


@dataclass
class RefinementConfig:
    hop_limit: int = 2
    degree_cap: int = 200
    hub_edge_cap: int = 50
    support_threshold: float = 0.55
    scorer_curated_weight: float = 0.35
    expansion_guide_top_k: int = 20
    curated_source_cap: int = 3


class SupportScorer:
    """Fixed support scorer s_e from the paper.

    ``curated_score`` is computed as min(1, n_curated / cap). ``text_score`` is
    expected to come from the PubMed BM25 + PubMedBERT pipeline described in
    Appendix B. For tiny examples or partially processed graphs, provenance
    scores are used as a deterministic fallback.
    """

    def __init__(self, curated_weight: float = 0.35, curated_source_cap: int = 3) -> None:
        self.curated_weight = curated_weight
        self.curated_source_cap = curated_source_cap

    def score(self, edge: Edge) -> float:
        n_curated = float(edge.metadata.get("n_curated_sources", 0.0))
        curated_score = float(edge.metadata.get(
            "curated_score",
            min(1.0, n_curated / max(self.curated_source_cap, 1)),
        ))
        text_score = edge.metadata.get("text_score")
        if text_score is None:
            text_score = max([item.score for item in edge.provenance], default=edge.support_score)
        text_score = float(text_score)
        score = self.curated_weight * curated_score + (1.0 - self.curated_weight) * text_score
        return max(0.0, min(1.0, score))


class EvidenceGraphRefiner:
    def __init__(self, config: RefinementConfig | None = None) -> None:
        self.config = config or RefinementConfig()
        self.scorer = SupportScorer(
            curated_weight=self.config.scorer_curated_weight,
            curated_source_cap=self.config.curated_source_cap,
        )

    def refine(
        self,
        disease_id: str,
        frequent_anchor_ids: Sequence[str],
        raw_nodes: Iterable[Node],
        raw_edges: Iterable[Edge],
    ) -> EvidenceGraph:
        nodes = {node.node_id: node for node in raw_nodes}
        edges = list(raw_edges)
        adjacency = self._build_adjacency(edges)

        disease_region = self._bfs_nodes({disease_id}, adjacency, self.config.hop_limit)
        candidate_edges = [
            edge for edge in edges if edge.head in disease_region and edge.tail in disease_region
        ]
        candidate_edges = self._restrict_to_anchor_component(
            disease_id=disease_id,
            anchor_ids=frequent_anchor_ids,
            edges=candidate_edges,
        )
        candidate_edges = self._apply_hub_control(candidate_edges)

        retained_edges = []
        for edge in candidate_edges:
            support_score = self.scorer.score(edge)
            if support_score < self.config.support_threshold:
                continue
            retained_edges.append(
                Edge(
                    edge_id=edge.edge_id,
                    head=edge.head,
                    relation=edge.relation,
                    tail=edge.tail,
                    support_score=support_score,
                    equiv_group_id=self._equivalence_key(edge),
                    provenance=list(edge.provenance),
                    metadata=dict(edge.metadata),
                )
            )

        equiv_groups = self._equivalence_groups(retained_edges)
        expansion_guide = self._expansion_guide(frequent_anchor_ids, retained_edges)
        used_nodes = {disease_id, *frequent_anchor_ids}
        for edge in retained_edges:
            used_nodes.update([edge.head, edge.tail])

        return EvidenceGraph(
            nodes={node_id: nodes[node_id] for node_id in used_nodes if node_id in nodes},
            edges={edge.edge_id: edge for edge in retained_edges},
            equiv_groups=equiv_groups,
            expansion_guide=expansion_guide,
            metadata={
                "target_disease_id": disease_id,
                "hop_limit": self.config.hop_limit,
                "support_threshold": self.config.support_threshold,
                "candidate_edge_count": len(candidate_edges),
                "retained_edge_count": len(retained_edges),
                "frequent_anchor_count": len(set(frequent_anchor_ids)),
            },
        )

    def _build_adjacency(self, edges: Sequence[Edge]) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.head].append(edge.tail)
            adjacency[edge.tail].append(edge.head)
        return adjacency

    def _bfs_nodes(self, starts: Set[str], adjacency: Dict[str, List[str]], max_hops: int) -> Set[str]:
        seen = set(starts)
        queue = deque((node_id, 0) for node_id in starts)
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in adjacency.get(node_id, []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
        return seen

    def _restrict_to_anchor_component(
        self,
        disease_id: str,
        anchor_ids: Sequence[str],
        edges: Sequence[Edge],
    ) -> List[Edge]:
        adjacency = self._build_adjacency(edges)
        anchors_in_region = {anchor for anchor in anchor_ids if anchor in adjacency}
        if not anchors_in_region:
            return [edge for edge in edges if disease_id in (edge.head, edge.tail)]
        reachable_from_anchors = self._bfs_nodes(anchors_in_region, adjacency, self.config.hop_limit + 1)
        reachable_from_disease = self._bfs_nodes({disease_id}, adjacency, self.config.hop_limit + 1)
        bridge_nodes = reachable_from_anchors & reachable_from_disease
        return [edge for edge in edges if edge.head in bridge_nodes and edge.tail in bridge_nodes]

    def _apply_hub_control(self, edges: Sequence[Edge]) -> List[Edge]:
        degree = Counter()
        for edge in edges:
            degree[edge.head] += 1
            degree[edge.tail] += 1

        kept: Dict[str, int] = defaultdict(int)
        output = []
        ranked = sorted(edges, key=self._prior_score, reverse=True)
        for edge in ranked:
            if degree[edge.head] > self.config.degree_cap and kept[edge.head] >= self.config.hub_edge_cap:
                continue
            if degree[edge.tail] > self.config.degree_cap and kept[edge.tail] >= self.config.hub_edge_cap:
                continue
            output.append(edge)
            kept[edge.head] += 1
            kept[edge.tail] += 1
        return output

    def _prior_score(self, edge: Edge) -> float:
        if "prior_score" in edge.metadata:
            return float(edge.metadata["prior_score"])
        if "text_score" in edge.metadata or "curated_score" in edge.metadata:
            return self.scorer.score(edge)
        return edge.support_score

    def _equivalence_key(self, edge: Edge) -> str:
        relation_category = edge.metadata.get("relation_category", edge.relation)
        return f"eq:{edge.head}|{relation_category}|{edge.tail}"

    def _equivalence_groups(self, edges: Sequence[Edge]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            groups[edge.equiv_group_id or edge.edge_id].append(edge.edge_id)
        return dict(groups)

    def _expansion_guide(self, anchor_ids: Sequence[str], edges: Sequence[Edge]) -> Dict[str, List[str]]:
        incident: Dict[str, List[Edge]] = defaultdict(list)
        for edge in edges:
            incident[edge.head].append(edge)
            incident[edge.tail].append(edge)

        guide: Dict[str, List[str]] = {}
        for anchor_id in anchor_ids:
            ranked = sorted(incident.get(anchor_id, []), key=lambda edge: edge.support_score, reverse=True)
            guide[anchor_id] = [edge.edge_id for edge in ranked[: self.config.expansion_guide_top_k]]
        return guide
