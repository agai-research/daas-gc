"""Graph clustering for sky network partitioning (DaaS-GC.tex, Algorithm 1).

Implements the SCAN-inspired partitioning: structural similarity (Ochiai
coefficient), epsilon-neighborhood, core/border/frontier/bridge/outlier
station classification, and a calibration routine that searches (eps, mu)
to approximate a target number of clusters k, as described in Section
"Impact of partitioning" of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from daasgc.sky_network import SkyNetwork


@dataclass
class ClusterResult:
    clusters: dict[int, set[str]]           # cluster_id -> station ids
    station_cluster: dict[str, int]         # station id -> cluster_id
    core: set[str] = field(default_factory=set)
    border: set[str] = field(default_factory=set)
    frontier: set[str] = field(default_factory=set)
    bridge: set[str] = field(default_factory=set)
    outlier: set[str] = field(default_factory=set)

    @property
    def k(self) -> int:
        return len(self.clusters)

    def cluster_of(self, station: str) -> int | None:
        return self.station_cluster.get(station)


def structural_similarity(sky: SkyNetwork, u: str, v: str) -> float:
    """Ochiai coefficient sigma(u, v), Eq. "eq:similarity"."""
    nu, nv = sky.neighbors(u), sky.neighbors(v)
    if not nu or not nv:
        return 0.0
    inter = len(nu & nv)
    return inter / math.sqrt(len(nu) * len(nv))


def epsilon_neighborhood(sky: SkyNetwork, u: str, eps: float) -> set[str]:
    """N_eps(u) = strong connections of u, Eq. "eq:strongconnection"."""
    return {v for v in sky.neighbors(u) if v != u and structural_similarity(sky, u, v) >= eps}


def cluster_sky_network(sky: SkyNetwork, eps: float, mu: int) -> ClusterResult:
    """Algorithm 1: clustering-based partitioning of the sky network.

    Steps: (1) neighbor aggregation, (2) similarity + core detection,
    (3) border detection, (4) station clustering, (5) bridge/outlier
    detection among the remaining unclustered stations, (6) frontier
    detection among already-clustered stations with a cross-cluster
    strong connection.
    """
    g = sky.graph
    core: set[str] = set()
    border: set[str] = set()
    strong_neighbors: dict[str, set[str]] = {}

    # Step 1-2: neighbor aggregation + core/border detection
    for rs in g.nodes:
        n_eps = epsilon_neighborhood(sky, rs, eps)
        strong_neighbors[rs] = n_eps
        if len(n_eps) >= mu:
            core.add(rs)

    for rs in g.nodes:
        if rs in core:
            continue
        if any(nb in core for nb in strong_neighbors[rs]):
            border.add(rs)

    # Step 3: assign each core station to a fresh cluster (seed)
    station_cluster: dict[str, int] = {}
    clusters: dict[int, set[str]] = {}
    next_cid = 0
    for rs in core:
        if rs in station_cluster:
            continue
        cid = next_cid
        next_cid += 1
        clusters[cid] = {rs}
        station_cluster[rs] = cid

    # Step 4: expand clusters by propagating cluster ids across strongly
    # connected core-core / core-border pairs (label propagation to a
    # fixed point, equivalent to the paper's "share cluster number with
    # neighbors" step).
    changed = True
    while changed:
        changed = False
        for rs in list(core) + list(border):
            if rs not in station_cluster:
                continue
            cid = station_cluster[rs]
            for nb in strong_neighbors[rs]:
                if nb in core or nb in border:
                    if nb not in station_cluster:
                        station_cluster[nb] = cid
                        clusters[cid].add(nb)
                        changed = True

    # Any border station connected to a core but not yet merged (should not
    # normally happen given the propagation above, kept for completeness).
    for rs in border:
        if rs not in station_cluster:
            for nb in strong_neighbors[rs]:
                if nb in station_cluster:
                    cid = station_cluster[nb]
                    station_cluster[rs] = cid
                    clusters[cid].add(rs)
                    break

    clustered = set(station_cluster.keys())

    # Step 5: bridge vs. outlier among unclustered stations.
    bridge: set[str] = set()
    outlier: set[str] = set()
    for rs in g.nodes:
        if rs in clustered:
            continue
        neighbor_clusters = {station_cluster[nb] for nb in g.neighbors(rs) if nb in station_cluster}
        if len(neighbor_clusters) >= 2:
            bridge.add(rs)
        else:
            outlier.add(rs)

    # Step 6: frontier detection - clustered station with a strong
    # cross-cluster edge (remains a member of its own cluster).
    frontier: set[str] = set()
    for rs in clustered:
        own_cid = station_cluster[rs]
        for nb in g.neighbors(rs):
            if nb in station_cluster and station_cluster[nb] != own_cid:
                if structural_similarity(sky, rs, nb) >= eps:
                    frontier.add(rs)
                    break

    return ClusterResult(
        clusters=clusters,
        station_cluster=station_cluster,
        core=core, border=border, frontier=frontier,
        bridge=bridge, outlier=outlier,
    )


def calibrate_for_target_k(sky: SkyNetwork, target_k: int,
                            eps_grid: list[float] | None = None,
                            mu_grid: list[int] | None = None) -> tuple[float, int, ClusterResult]:
    """Sweep (eps, mu) and keep the configuration closest to target_k clusters.

    Mirrors the calibration procedure of Section "Impact of partitioning":
    the number of clusters k is not a direct input, it results from (eps,
    mu), so we search for the pair producing the closest non-trivial k.
    """
    if eps_grid is None:
        eps_grid = [round(x, 2) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    if mu_grid is None:
        n = sky.n_stations()
        mu_grid = sorted(set([2, 3, 4, 5, 6, 8, 10, max(2, n // 20)]))

    best = None
    best_gap = math.inf
    for eps in eps_grid:
        for mu in mu_grid:
            result = cluster_sky_network(sky, eps, mu)
            non_trivial = sum(1 for c in result.clusters.values() if len(c) > 1)
            if non_trivial == 0:
                continue
            gap = abs(non_trivial - target_k)
            if gap < best_gap:
                best_gap = gap
                best = (eps, mu, result)
            if gap == 0:
                return best
    if best is None:
        # Fall back to a single, coarse configuration (k=1 equivalent).
        eps, mu = 0.1, 2
        return eps, mu, cluster_sky_network(sky, eps, mu)
    return best


def build_hypergraph(sky: SkyNetwork, result: ClusterResult) -> nx.Graph:
    """Build the sky network hyper-graph G^f (Definition "Sky network hyper
    graph"): a sub-network containing frontier, bridge, and outlier
    stations, connected by hyper-edges representing inter-cluster
    connectivity.
    """
    hg = nx.Graph()
    special = result.frontier | result.bridge | result.outlier
    for rs in special:
        hg.add_node(rs, cluster=result.cluster_of(rs))
    for u, v, data in sky.graph.edges(data=True):
        if u in special and v in special:
            hg.add_edge(u, v, **data)
    return hg
