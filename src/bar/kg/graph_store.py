"""In-memory refined evidence graph store."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bar.schemas import Edge, EvidenceGraph, Node, unique_preserve_order


class InMemoryGraphStore:
    """Fast lookup layer for refined evidence graphs.

    The store treats graph traversal as undirected by default because medical
    evidence paths often need to move from an EHR anchor toward a disease target
    even when the source KG edge direction points the other way.
    """

    def __init__(self, graph: EvidenceGraph, undirected: bool = True) -> None:
        errors = graph.validate()
        if errors:
            raise ValueError("Invalid evidence graph:\n" + "\n".join(errors))
        self.graph = graph
        self.undirected = undirected
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in graph.edges.values():
            self._adjacency[edge.head].append(edge.edge_id)
            if undirected:
                self._adjacency[edge.tail].append(edge.edge_id)
        for node_id in self._adjacency:
            self._adjacency[node_id].sort(
                key=lambda edge_id: graph.edges[edge_id].support_score,
                reverse=True,
            )

    @classmethod
    def load(cls, path: str | Path, undirected: bool = True) -> "InMemoryGraphStore":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(EvidenceGraph.from_dict(data), undirected=undirected)

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.graph.to_dict(), handle, indent=2, sort_keys=True)

    def node(self, node_id: str) -> Optional[Node]:
        return self.graph.nodes.get(node_id)

    def edge(self, edge_id: str) -> Optional[Edge]:
        return self.graph.edges.get(edge_id)

    def neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        top_k: Optional[int] = None,
        min_support: float = 0.0,
    ) -> List[Edge]:
        edge_ids = self._adjacency.get(node_id, [])
        edges = []
        for edge_id in edge_ids:
            edge = self.graph.edges[edge_id]
            if relation is not None and edge.relation != relation:
                continue
            if edge.support_score < min_support:
                continue
            edges.append(edge)
        if top_k is not None:
            return edges[:top_k]
        return edges

    def frontier(self, node_id: str, top_k: int = 5, min_support: float = 0.0) -> List[Edge]:
        indexed_ids = self.graph.expansion_guide.get(node_id, [])
        if indexed_ids:
            edges = [
                self.graph.edges[edge_id]
                for edge_id in indexed_ids
                if edge_id in self.graph.edges and self.graph.edges[edge_id].support_score >= min_support
            ]
            return sorted(edges, key=lambda edge: edge.support_score, reverse=True)[:top_k]
        return self.neighbors(node_id, top_k=top_k, min_support=min_support)

    def dedupe_edges(self, edge_ids: Iterable[str]) -> List[str]:
        """Keep the highest-support edge per equivalence group."""

        best_by_group: Dict[str, Edge] = {}
        for edge_id in edge_ids:
            edge = self.graph.edges.get(edge_id)
            if edge is None:
                continue
            group_id = edge.equiv_group_id or edge.edge_id
            current = best_by_group.get(group_id)
            if current is None or edge.support_score > current.support_score:
                best_by_group[group_id] = edge
        ranked = sorted(best_by_group.values(), key=lambda edge: edge.support_score, reverse=True)
        return [edge.edge_id for edge in ranked]

    def endpoint_ids(self, edge: Edge, from_node: Optional[str] = None) -> List[str]:
        endpoints = [edge.head, edge.tail]
        if from_node is not None:
            endpoints = [node_id for node_id in endpoints if node_id != from_node]
        return unique_preserve_order(endpoints)
