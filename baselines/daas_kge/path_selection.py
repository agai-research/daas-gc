"""Skyway path selection over the embedding subspace (DaaS_KGE-1.pdf,
Algorithm 2). Restricted to the high-proximity candidate stations, a
shortest-path search (equivalent to the paper's `delivery_paths()`
Dijkstra routine) produces the candidate path, scored with the paper's
weighted criteria (flight time, cost, number of stations).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from daasgc.path_search import path_score
from daasgc.sky_network import SkyNetwork


@dataclass
class KGEPathResult:
    stations: list[str]
    score: float


def select_skyway_path(sky: SkyNetwork, source: str, target: str,
                        candidate_stations: list[str]) -> KGEPathResult | None:
    sub = sky.graph.subgraph(candidate_stations)
    if source not in sub or target not in sub:
        sub = sky.graph  # fall back to the full graph if the subspace disconnects
    try:
        path = nx.dijkstra_path(sub, source, target, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            path = nx.dijkstra_path(sky.graph, source, target, weight="distance")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    return KGEPathResult(stations=path, score=path_score(sky, path))
