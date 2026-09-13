"""High-level DaaS-GC / DaaS-HGC pipelines.

Ties together: (1) clustering-based partitioning (clustering.py), (2)
delivery path search - either the DaaS-GC "try every frontier exit"
strategy (path_search.py) or the DaaS-HGC hyper-graph-first strategy
(hypergraph.py), and (3) drone formation selection (composition.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.clustering import ClusterResult, build_hypergraph, calibrate_for_target_k, cluster_sky_network
from daasgc.composition import Formation, select_drone_formation
from daasgc.hypergraph import hgc_path_search
from daasgc.path_search import PathResult, delivery_path_search
from daasgc.sky_network import SkyNetwork


@dataclass
class DaaSResult:
    path: PathResult
    formation: Formation | None
    cluster_result: ClusterResult
    eps: float
    mu: int


def _prepare_clusters(sky: SkyNetwork, target_k: int | None, eps: float | None,
                       mu: int | None) -> tuple[float, int, ClusterResult]:
    if target_k is not None:
        return calibrate_for_target_k(sky, target_k)
    eps = eps if eps is not None else 0.3
    mu = mu if mu is not None else 3
    return eps, mu, cluster_sky_network(sky, eps, mu)


def run_daas_gc(sky: SkyNetwork, source: str, target: str, items: list[dict],
                weights: dict[str, float], target_k: int | None = 5,
                eps: float | None = None, mu: int | None = None) -> DaaSResult:
    eps, mu, cluster_result = _prepare_clusters(sky, target_k, eps, mu)
    hypergraph = build_hypergraph(sky, cluster_result)
    path = delivery_path_search(sky, source, target, cluster_result, hypergraph)
    formation = select_drone_formation(sky, path.stations, items, weights)
    return DaaSResult(path=path, formation=formation, cluster_result=cluster_result, eps=eps, mu=mu)


def run_daas_hgc(sky: SkyNetwork, source: str, target: str, items: list[dict],
                 weights: dict[str, float], target_k: int | None = 5,
                 eps: float | None = None, mu: int | None = None) -> DaaSResult:
    eps, mu, cluster_result = _prepare_clusters(sky, target_k, eps, mu)
    hypergraph = build_hypergraph(sky, cluster_result)
    path = hgc_path_search(sky, source, target, cluster_result, hypergraph)
    formation = select_drone_formation(sky, path.stations, items, weights)
    return DaaSResult(path=path, formation=formation, cluster_result=cluster_result, eps=eps, mu=mu)
