"""Delivery path search (DaaS-GC.tex, Algorithm 2).

Given source/target stations and the sky network partitioning, finds the
optimal delivery path across one of three scenarios: single-cluster,
two-cluster, or multi-cluster. Local paths inside a cluster are found with
an admissible A* search; the path-level scoring function (Eq.
"eq:path-score") is applied to select among candidate local paths and is
kept strictly separate from the drone-formation QoS score used later in
composition.py (two-phase design described in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from daasgc.clustering import ClusterResult
from daasgc.sky_network import SkyNetwork


@dataclass
class PathResult:
    stations: list[str]
    score: float
    scenario: str


def path_score(sky: SkyNetwork, stations: list[str]) -> float:
    """Eq. "eq:path-score": score(P) = r * (sum(dist) + sum(wait) + sum(cost))."""
    if len(stations) < 1:
        return float("inf")
    r = len(stations)
    total_dist = sum(sky.segment_distance(stations[i], stations[i + 1])
                      for i in range(len(stations) - 1))
    total_wait = sum(sky.waiting_time(rs) for rs in stations)
    total_cost = sum(sky.charging_cost(rs) for rs in stations)
    return r * (total_dist + total_wait + total_cost)


def _admissible_heuristic(sky: SkyNetwork, node: str, target: str, min_unit_cost: float) -> float:
    """Lower bound on the remaining path score: straight-line proxy (here,
    the minimum edge distance in the graph) scaled by the minimum per-segment
    cost, guaranteeing h(rs) <= h*(rs) as required for A* admissibility.
    """
    try:
        hops = nx.shortest_path_length(sky.graph, node, target)
    except nx.NetworkXNoPath:
        return float("inf")
    return hops * min_unit_cost


def a_star_search(sky: SkyNetwork, source: str, target: str,
                   allowed_stations: set[str] | None = None) -> PathResult | None:
    """A* search restricted to `allowed_stations` (a single cluster, or the
    whole graph for DaaS-SG), minimizing the additive per-segment cost used
    in Eq. "eq:path-score" (distance + waiting + charging).
    """
    if allowed_stations is None:
        sub = sky.graph
    else:
        sub = sky.graph.subgraph(allowed_stations)
    if source not in sub or target not in sub:
        return None

    min_unit_cost = min((d["distance"] for _, _, d in sub.edges(data=True)), default=1.0)
    min_unit_cost = max(min_unit_cost, 1e-6)

    def weight(u, v, data):
        return data["distance"] + sky.waiting_time(v) + sky.charging_cost(v)

    def heuristic(u, v):
        return _admissible_heuristic(sky, u, v, min_unit_cost)

    try:
        path = nx.astar_path(sub, source, target, heuristic=heuristic, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    return PathResult(stations=path, score=path_score(sky, path), scenario="local")


def _traversed_cluster_sequence(cluster_result: ClusterResult, source_cid: int, target_cid: int,
                                 hypergraph: nx.Graph) -> list[int]:
    """Find the sequence of cluster ids connecting source_cid to target_cid
    through the frontier/bridge hyper-graph (breadth-first over clusters).
    """
    cluster_adjacency: dict[int, set[int]] = {cid: set() for cid in cluster_result.clusters}
    for u, v in hypergraph.edges():
        cu, cv = hypergraph.nodes[u].get("cluster"), hypergraph.nodes[v].get("cluster")
        if cu is not None and cv is not None and cu != cv:
            cluster_adjacency.setdefault(cu, set()).add(cv)
            cluster_adjacency.setdefault(cv, set()).add(cu)

    visited = {source_cid}
    queue = [[source_cid]]
    while queue:
        path = queue.pop(0)
        last = path[-1]
        if last == target_cid:
            return path
        for nb in cluster_adjacency.get(last, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(path + [nb])
    return [source_cid, target_cid]


def _frontier_candidates(sky: SkyNetwork, cluster_result: ClusterResult, cid: int,
                          next_cid: int | None) -> list[str]:
    """Frontier (or bridge) stations of cluster `cid` reachable towards `next_cid`."""
    cluster_stations = cluster_result.clusters.get(cid, set())
    candidates = [rs for rs in cluster_stations if rs in cluster_result.frontier]
    if next_cid is not None:
        next_stations = cluster_result.clusters.get(next_cid, set())
        filtered = [rs for rs in candidates
                    if any(nb in next_stations or nb in cluster_result.bridge
                           for nb in sky.graph.neighbors(rs))]
        if filtered:
            return filtered
    return candidates or list(cluster_stations)


def delivery_path_search(sky: SkyNetwork, source: str, target: str,
                          cluster_result: ClusterResult, hypergraph: nx.Graph) -> PathResult:
    """Algorithm 2: single-cluster / two-cluster / multi-cluster delivery
    path search, driven by the sky network partitioning.
    """
    cs = cluster_result.cluster_of(source)
    ct = cluster_result.cluster_of(target)

    # Outlier/bridge source or target: connect directly to the nearest
    # frontier/bridge station, then proceed as usual.
    if cs is None:
        cs = _nearest_cluster(sky, cluster_result, source)
    if ct is None:
        ct = _nearest_cluster(sky, cluster_result, target)

    if cs == ct:
        result = a_star_search(sky, source, target, cluster_result.clusters.get(cs, {source, target}))
        if result is None:
            result = a_star_search(sky, source, target, None)
        result.scenario = "single-cluster"
        return result

    seq = _traversed_cluster_sequence(cluster_result, cs, ct, hypergraph)
    full_path: list[str] = []
    current_source = source
    for i, cid in enumerate(seq):
        is_last = i == len(seq) - 1
        cluster_stations = cluster_result.clusters.get(cid, {current_source})
        cluster_stations = cluster_stations | {current_source}
        if is_last:
            exits = [target]
        else:
            exits = _frontier_candidates(sky, cluster_result, cid, seq[i + 1])

        best_local = None
        for exit_station in exits:
            local = a_star_search(sky, current_source, exit_station, cluster_stations | {exit_station})
            if local is None:
                continue
            if best_local is None or local.score < best_local.score:
                best_local = local
        if best_local is None:
            # Fall back to a direct hop via the whole graph if the cluster
            # view fails to connect (keeps the search complete).
            best_local = a_star_search(sky, current_source, exits[0] if exits else target, None)
        if best_local is None:
            continue
        if full_path and full_path[-1] == best_local.stations[0]:
            full_path.extend(best_local.stations[1:])
        else:
            full_path.extend(best_local.stations)
        current_source = full_path[-1]

    if not full_path or full_path[-1] != target:
        direct = a_star_search(sky, current_source, target, None)
        if direct is not None:
            full_path.extend(direct.stations[1:])

    scenario = "two-cluster" if len(seq) == 2 else "multi-cluster"
    return PathResult(stations=full_path, score=path_score(sky, full_path), scenario=scenario)


def _nearest_cluster(sky: SkyNetwork, cluster_result: ClusterResult, station: str) -> int:
    """For an outlier/bridge endpoint, attach to the cluster of its closest
    clustered neighbor (direct linkage rule described for outlier stations).
    """
    for nb in sky.graph.neighbors(station):
        cid = cluster_result.cluster_of(nb)
        if cid is not None:
            return cid
    # Isolated station with no clustered neighbor: create an ad hoc singleton.
    return -1
