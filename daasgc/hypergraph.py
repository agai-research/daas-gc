"""DaaS-HGC: hyper-graph abstraction variant of the delivery path search.

Unlike DaaS-GC (path_search.delivery_path_search), which evaluates the
local path to *every* candidate exit frontier station of a traversed
cluster, DaaS-HGC first picks the frontier/bridge route through the
sky-network hyper-graph G^f (Definition "Sky network hyper graph") and
only then runs the local A* search along that fixed sequence of frontier
stations. This is cheaper for large numbers of clusters but can miss a
slightly better frontier choice, matching the trade-off discussed in
Section "Impact of partitioning" of the paper.
"""

from __future__ import annotations

import networkx as nx

from daasgc.clustering import ClusterResult
from daasgc.path_search import PathResult, a_star_search, path_score
from daasgc.sky_network import SkyNetwork


def hgc_path_search(sky: SkyNetwork, source: str, target: str,
                     cluster_result: ClusterResult, hypergraph: nx.Graph) -> PathResult:
    cs = cluster_result.cluster_of(source)
    ct = cluster_result.cluster_of(target)

    if cs == ct and cs is not None:
        local = a_star_search(sky, source, target, cluster_result.clusters.get(cs))
        if local is None:
            local = a_star_search(sky, source, target, None)
        local.scenario = "single-cluster"
        return local

    # Build an augmented hyper-graph including source/target so a single
    # shortest-path call selects the frontier route through G^f.
    hg = hypergraph.copy()
    hg.add_node(source, cluster=cs)
    hg.add_node(target, cluster=ct)
    for rs in (source, target):
        for nb in sky.graph.neighbors(rs):
            if nb in hg:
                hg.add_edge(rs, nb, **sky.graph.edges[rs, nb])

    try:
        frontier_route = nx.shortest_path(hg, source, target, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        frontier_route = [source, target]

    full_path: list[str] = []
    for i in range(len(frontier_route) - 1):
        u, v = frontier_route[i], frontier_route[i + 1]
        cu = cluster_result.cluster_of(u)
        allowed = cluster_result.clusters.get(cu, {u, v}) | {u, v} if cu is not None else None
        local = a_star_search(sky, u, v, allowed)
        if local is None:
            local = a_star_search(sky, u, v, None)
        if local is None:
            continue
        if full_path and full_path[-1] == local.stations[0]:
            full_path.extend(local.stations[1:])
        else:
            full_path.extend(local.stations)

    if not full_path:
        full_path = frontier_route

    scenario = "two-cluster" if len(frontier_route) <= 2 else "multi-cluster"
    return PathResult(stations=full_path, score=path_score(sky, full_path), scenario=scenario)
