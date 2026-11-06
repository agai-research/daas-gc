"""DaaS-HGC: hyper-graph abstraction variant of the delivery path search.

The gateway route is chosen first on the sky-network hyper-graph G^f
(gateways, bridges and outliers; intra-cluster virtual edges weighted by the
straight-line distance), then local A* searches are run only between
consecutive gateways of that route. Cheaper than DaaS-GC's exhaustive
exploration of exit gateways, at the price of possibly sub-optimal local
choices (the virtual edges only approximate intra-cluster costs).
"""

from __future__ import annotations

import networkx as nx

from daasgc.clustering import _edge_weight
from daasgc.path_search import (Partition, PathResult, a_star_search, additive_cost, _full_search,
                                path_score, region_search)
from daasgc.sky_network import SkyNetwork


def hgc_path_search(sky: SkyNetwork, source: str, target: str, part: Partition) -> PathResult:
    cs, ct = part.home(source), part.home(target)
    if cs is None or ct is None:
        return _full_search(sky, source, target, 0)
    if cs == ct:
        res = region_search(sky, part, cs, source, target)
        if res is None:
            return _full_search(sky, source, target, 1)
        res.scenario = "single-cluster"
        return res

    hg = part.hypergraph
    added = []
    for rs, c in ((source, cs), (target, ct)):
        if rs in hg:
            continue
        hg.add_node(rs, cluster=part.clusters.cluster_of(rs))
        added.append(rs)
        for g in [n for n, d in hg.nodes(data=True) if d.get("cluster") == c and n != rs]:
            hg.add_edge(rs, g, weight=_edge_weight(sky, rs, g, sky.euclid(rs, g)), virtual=True, cluster=c)
        for nb in sky.graph.neighbors(rs):
            if nb in hg and nb != rs and not hg.has_edge(rs, nb) and part.clusters.cluster_of(nb) != c:
                hg.add_edge(rs, nb, weight=_edge_weight(sky, rs, nb, sky.segment_distance(rs, nb)), virtual=False)
    try:
        route = nx.dijkstra_path(hg, source, target, weight="weight")
        virtual = [hg.edges[route[i], route[i + 1]].get("virtual", False) for i in range(len(route) - 1)]
        clusters_of = [hg.edges[route[i], route[i + 1]].get("cluster") for i in range(len(route) - 1)]
    except nx.NetworkXNoPath:
        route = None
    finally:
        hg.remove_nodes_from(added)
    if route is None:
        return _full_search(sky, source, target, 0)

    full, calls, fallback = [source], 0, False
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if virtual[i]:
            local = region_search(sky, part, clusters_of[i], u, v)
            calls += 1
            if local is None:
                local, fallback = _full_search(sky, u, v, 0), True
            full += local.stations[1:]
        else:
            full.append(v)
    n_clusters = len({c for c in clusters_of if c is not None})
    scenario = "fallback" if fallback else ("two-cluster" if n_clusters <= 2 else "multi-cluster")
    return PathResult(full, path_score(sky, full), scenario, additive_cost(sky, full), calls, fallback)
