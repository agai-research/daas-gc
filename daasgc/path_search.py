"""Delivery path search (Algorithm 2) on a partitioned sky network.

Local paths are found with A* (edge cost = segment distance + waiting time +
charging cost of the reached station; heuristic = straight-line distance to
the target, which never exceeds the remaining cost, hence admissible and
consistent). Candidate paths are compared with the path score of
Eq. path-score. DaaS-GC evaluates the local path between *every* entry and
*every* exit gateway of each traversed cluster ("processing all the entry
and exit frontier stations, which results in a combinatorial situation",
Section 6.3.4) and keeps the cheapest chain of local paths (dynamic
programming over the traversed clusters); DaaS-HGC (hypergraph.py) first
chooses the gateway route on the hyper-graph, then runs local searches only
along it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from daasgc.clustering import ClusterResult, build_hypergraph
from daasgc.sky_network import SkyNetwork

# True: every entry x every exit of each traversed cluster, cheapest chain (Section 6.3.4).
# False: Algorithm 2 pseudo-code reading - from the single current entry, keep only the best
# exit/next entry before moving to the next cluster (greedy, cluster by cluster).
GC_ALL_ENTRIES = True


@dataclass
class PathResult:
    stations: list[str]
    score: float
    scenario: str
    additive_cost: float = 0.0
    astar_calls: int = 0
    fallback: bool = False


def path_score(sky: SkyNetwork, stations: list[str]) -> float:
    """Eq. path-score: r * (sum dist + sum t_w + sum cost), r = |P|."""
    if not stations:
        return float("inf")
    return len(stations) * (sky.path_distance(stations)
                            + sum(sky.waiting_time(s) for s in stations)
                            + sum(sky.charging_cost(s) for s in stations))


def step_cost(sky: SkyNetwork, u: str, v: str) -> float:
    return sky.segment_distance(u, v) + sky.waiting_time(v) + sky.charging_cost(v)


def additive_cost(sky: SkyNetwork, stations: list[str]) -> float:
    return sum(step_cost(sky, stations[i], stations[i + 1]) for i in range(len(stations) - 1))


def a_star_search(sky: SkyNetwork, source: str, target: str,
                  allowed: set[str] | None = None, graph: nx.Graph | None = None) -> PathResult | None:
    if graph is not None:
        g = graph
    elif allowed is None or len(allowed) >= sky.n_stations():
        g = sky.graph
    else:
        g = sky.graph.subgraph(allowed)
    if source not in g or target not in g:
        return None
    if source == target:
        return PathResult([source], path_score(sky, [source]), "local", 0.0, 1)
    try:
        path = nx.astar_path(g, source, target,
                             heuristic=lambda a, b: sky.euclid(a, b),
                             weight=lambda u, v, d: d["distance"] + sky.waiting_time(v) + sky.charging_cost(v))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return PathResult(path, path_score(sky, path), "local", additive_cost(sky, path), 1)


@dataclass
class Partition:
    """Pre-processed view of a partitioned network (built once, reused by queries)."""

    clusters: ClusterResult
    regions: dict[int, set[str]] = field(default_factory=dict)
    region_graphs: dict[int, nx.Graph] = field(default_factory=dict)
    attach: dict[str, int | None] = field(default_factory=dict)
    cluster_graph: nx.Graph = field(default_factory=nx.Graph)
    hypergraph: nx.Graph = field(default_factory=nx.Graph)

    @property
    def k(self) -> int:
        return self.clusters.k

    def home(self, station: str) -> int | None:
        c = self.clusters.cluster_of(station)
        return c if c is not None else self.attach.get(station)


def prepare_partition(sky: SkyNetwork, clusters: ClusterResult) -> Partition:
    p = Partition(clusters=clusters)
    g = sky.graph
    lab = clusters.station_cluster
    unclustered = [u for u in g.nodes if u not in lab]
    for c, members in clusters.clusters.items():
        p.regions[c] = set(members)
    for u in unclustered:
        cl = {lab[v] for v in g.neighbors(u) if v in lab}
        for c in cl:
            p.regions[c].add(u)
        nbrs = sorted((sky.euclid(u, v), v) for v in g.neighbors(u) if v in lab)
        p.attach[u] = lab[nbrs[0][1]] if nbrs else None
    cent = {}
    for c, members in clusters.clusters.items():
        xs = [g.nodes[m]["x"] for m in members]
        ys = [g.nodes[m]["y"] for m in members]
        cent[c] = (sum(xs) / len(xs), sum(ys) / len(ys))
        p.cluster_graph.add_node(c)

    def link(a, b):
        if a != b and not p.cluster_graph.has_edge(a, b):
            d = ((cent[a][0] - cent[b][0]) ** 2 + (cent[a][1] - cent[b][1]) ** 2) ** 0.5
            p.cluster_graph.add_edge(a, b, weight=d)

    for u, v in g.edges:
        cu, cv = lab.get(u), lab.get(v)
        if cu is not None and cv is not None:
            link(cu, cv)
    for u in unclustered:
        cl = sorted({lab[v] for v in g.neighbors(u) if v in lab})
        for i, a in enumerate(cl):
            for b in cl[i + 1:]:
                link(a, b)
    for c, members in p.regions.items():           # materialised once (pre-processing)
        p.region_graphs[c] = g if len(members) >= g.number_of_nodes() else g.subgraph(members).copy()
    if clusters.k > 1:
        p.hypergraph = build_hypergraph(sky, clusters)
    return p


def region_search(sky: SkyNetwork, part: "Partition", c: int, source: str, target: str) -> PathResult | None:
    """Local A* inside region c (augmented with the endpoints if they lie outside it)."""
    g = part.region_graphs[c]
    if source in g and target in g:
        return a_star_search(sky, source, target, graph=g)
    return a_star_search(sky, source, target, part.regions[c] | {source, target})


def _full_search(sky, source, target, calls, scenario="fallback") -> PathResult:
    res = a_star_search(sky, source, target, None)
    if res is None:
        raise ValueError(f"No path between {source} and {target}")
    res.scenario, res.astar_calls, res.fallback = scenario, calls + 1, scenario == "fallback"
    return res


def delivery_path_search(sky: SkyNetwork, source: str, target: str, part: Partition) -> PathResult:
    """Algorithm 2 (DaaS-GC): single / two / multi-cluster path search."""
    cs, ct = part.home(source), part.home(target)
    if cs is None or ct is None:
        return _full_search(sky, source, target, 0)
    if cs == ct:
        res = region_search(sky, part, cs, source, target)
        if res is None:
            return _full_search(sky, source, target, 1)
        res.scenario = "single-cluster"
        return res
    try:
        seq = nx.dijkstra_path(part.cluster_graph, cs, ct, weight="weight")
    except nx.NetworkXNoPath:
        return _full_search(sky, source, target, 0)
    clusters = part.clusters.clusters
    calls = 0
    # layer: entry station of the current cluster -> (cumulative additive cost, path so far)
    layer: dict[str, tuple[float, list[str]]] = {source: (0.0, [source])}
    for i, cid in enumerate(seq[:-1]):
        nxt = clusters[seq[i + 1]]
        region = part.regions[cid]
        exits = sorted(x for x in region if any(e in nxt for e in sky.graph.neighbors(x)))
        new_layer: dict[str, tuple[float, list[str]]] = {}
        for entry in sorted(layer):                      # every entry ...
            base_cost, base_path = layer[entry]
            for x in exits:                              # ... x every exit of the cluster
                if x == entry:
                    local_cost, local_path = 0.0, [entry]
                else:
                    local = region_search(sky, part, cid, entry, x)
                    calls += 1
                    if local is None:
                        continue
                    local_cost, local_path = local.additive_cost, local.stations
                for e in sorted(sky.graph.neighbors(x)):
                    if e not in nxt or e in base_path or e in local_path:
                        continue
                    c = base_cost + local_cost + step_cost(sky, x, e)
                    if e not in new_layer or c < new_layer[e][0]:
                        new_layer[e] = (c, base_path + local_path[1:] + [e])
        if not new_layer:
            break
        if not GC_ALL_ENTRIES:
            e_best = min(new_layer, key=lambda e: (new_layer[e][0], e))
            new_layer = {e_best: new_layer[e_best]}
        layer = new_layer
    else:
        best = None
        for entry in sorted(layer):
            base_cost, base_path = layer[entry]
            if entry == target:
                cand = (base_cost, base_path)
            else:
                last = region_search(sky, part, ct, entry, target)
                calls += 1
                if last is None:
                    continue
                cand = (base_cost + last.additive_cost, base_path + last.stations[1:])
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            full = best[1]
            scenario = "two-cluster" if len(seq) == 2 else "multi-cluster"
            return PathResult(full, path_score(sky, full), scenario, additive_cost(sky, full), calls, False)
    res = _full_search(sky, source, target, calls)
    return res
