"""Graph clustering for sky network partitioning (Algorithm 1).

SCAN semantics: sigma = Ochiai coefficient over structural neighbourhoods
N(rs) = neighbours(rs) U {rs} (Eq. similarity); N_eps(u) = {v in N(u) |
sigma(u, v) >= eps} (Eq. strongconnection, includes u itself); u is a core
station if |N_eps(u)| >= mu (Eq. core). A cluster is grown from an unlabelled
core by repeatedly adding the eps-neighbours of every core it contains, so
strongly-connected cores always end up in the same cluster; non-core
stations reached this way are border stations. Unclustered stations are
bridges (neighbours in >= 2 clusters) or outliers.

Functional resemblance (Def. Cluster: a cluster groups stations that are
"structurally similar (e.g., types of pads, charging voltage, etc.) and
closely connected"; Sections 6.3.1-7.3.3 attribute DaaS-GC's results to
regions that are homogeneous in charging characteristics): a segment is a
strong connection only if sigma(u, v) >= eps AND both stations share the same
functional profile (pad type, the attribute that decides drone
compatibility). Set FUNCTIONAL_GATE = False for purely structural clustering.

Frontier stations follow the paper's definition (clustered station with a
strong cross-cluster edge). Because SCAN separates clusters precisely along
weak edges, that set is usually tiny, so path search uses *gateway*
stations: clustered stations with any edge leaving their cluster (to another
cluster or to a bridge/outlier). Both sets are reported.

k = 1 is the degenerate partition: the whole network is a single region, no
frontier/gateway/bridge/outlier stations; DaaS-GC and DaaS-HGC then run the
same full-graph A* search as DaaS-SG.

The (eps, mu) calibration that approximates a target k is a pre-processing
step (`calibration_sweep`, `calibrate_for_target_k`), run once per network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from daasgc.sky_network import SkyNetwork

EPS_GRID = [round(0.30 + 0.01 * i, 2) for i in range(66)]          # 0.30 .. 0.95
MU_GRID = list(range(2, 21))
FUNCTIONAL_GATE = True


@dataclass
class ClusterResult:
    clusters: dict[int, set[str]]
    station_cluster: dict[str, int]
    eps: float | None = None
    mu: int | None = None
    core: set[str] = field(default_factory=set)
    border: set[str] = field(default_factory=set)
    frontier: set[str] = field(default_factory=set)
    gateway: set[str] = field(default_factory=set)
    bridge: set[str] = field(default_factory=set)
    outlier: set[str] = field(default_factory=set)

    @property
    def k(self) -> int:
        return len(self.clusters)

    def cluster_of(self, station: str) -> int | None:
        return self.station_cluster.get(station)

    def summary(self) -> dict:
        return {"k": self.k, "eps": self.eps, "mu": self.mu,
                "sizes": sorted((len(c) for c in self.clusters.values()), reverse=True),
                "core": len(self.core), "border": len(self.border),
                "frontier_paper_def": len(self.frontier), "gateway": len(self.gateway),
                "bridge": len(self.bridge), "outlier": len(self.outlier)}


def structural_similarity(sky: SkyNetwork, u: str, v: str) -> float:
    nu, nv = sky.neighbors(u), sky.neighbors(v)
    return len(nu & nv) / math.sqrt(len(nu) * len(nv))


def similarity_table(sky: SkyNetwork) -> dict[str, dict[str, float]]:
    """sigma(u, v) for every segment, computed once per network and cached."""
    if "sim" not in sky._cache:
        nb = {u: sky.neighbors(u) for u in sky.graph.nodes}
        sim = {u: {} for u in sky.graph.nodes}
        for u, v in sky.graph.edges:
            s = len(nb[u] & nb[v]) / math.sqrt(len(nb[u]) * len(nb[v]))
            sim[u][v] = sim[v][u] = s
        sky._cache["sim"] = sim
    return sky._cache["sim"]


def _scan(order: list[str], strong: dict[str, list[str]], mu: int):
    core = {u for u in order if len(strong[u]) + 1 >= mu}          # +1: u itself is in N_eps(u)
    label: dict[str, int] = {}
    cid = 0
    for seed in order:
        if seed not in core or seed in label:
            continue
        label[seed] = cid
        stack = [seed]
        while stack:
            x = stack.pop()
            for y in strong[x]:
                if y not in label:
                    label[y] = cid
                    if y in core:
                        stack.append(y)
        cid += 1
    return core, label, cid


def functionally_similar(sky: SkyNetwork, u: str, v: str) -> bool:
    return (not FUNCTIONAL_GATE) or sky.functional_profile(u) == sky.functional_profile(v)


def _strong_adjacency(sky: SkyNetwork, eps: float) -> dict[str, list[str]]:
    sim = similarity_table(sky)
    return {u: [v for v in sky.graph.neighbors(u) if sim[u][v] >= eps and functionally_similar(sky, u, v)]
            for u in sky.graph.nodes}


def cluster_sky_network(sky: SkyNetwork, eps: float, mu: int) -> ClusterResult:
    order = list(sky.graph.nodes)
    strong = _strong_adjacency(sky, eps)
    core, label, n = _scan(order, strong, mu)
    clusters = {c: set() for c in range(n)}
    for u in order:
        if u in label:
            clusters[label[u]].add(u)
    res = ClusterResult(clusters=clusters, station_cluster=label, eps=eps, mu=mu, core=core,
                        border={u for u in label if u not in core})
    sim = similarity_table(sky)
    g = sky.graph
    for u in order:
        if u in label:
            for v in g.neighbors(u):
                if label.get(v) != label[u]:
                    res.gateway.add(u)
                    if v in label and sim[u][v] >= eps and functionally_similar(sky, u, v):
                        res.frontier.add(u)
        else:
            if len({label[v] for v in g.neighbors(u) if v in label}) >= 2:
                res.bridge.add(u)
            else:
                res.outlier.add(u)
    return res


def single_cluster(sky: SkyNetwork) -> ClusterResult:
    nodes = set(sky.graph.nodes)
    return ClusterResult(clusters={0: nodes}, station_cluster={u: 0 for u in sky.graph.nodes})


def calibration_sweep(sky: SkyNetwork, eps_grid=None, mu_grid=None) -> list[dict]:
    """Number of clusters and of clustered stations for every (eps, mu) pair."""
    key = ("sweep", FUNCTIONAL_GATE, tuple(eps_grid or EPS_GRID), tuple(mu_grid or MU_GRID))
    if key in sky._cache:
        return sky._cache[key]
    order = list(sky.graph.nodes)
    rows = []
    for eps in eps_grid or EPS_GRID:
        strong = _strong_adjacency(sky, eps)
        for mu in mu_grid or MU_GRID:
            _, label, n = _scan(order, strong, mu)
            rows.append({"eps": eps, "mu": mu, "k": n, "clustered": len(label)})
    sky._cache[key] = rows
    return rows


def calibrate_for_target_k(sky: SkyNetwork, target_k: int, sweep: list[dict] | None = None):
    """Pick the (eps, mu) whose number of clusters is closest to target_k
    (ties: most clustered stations, then smallest mu, then smallest eps)."""
    sweep = sweep or calibration_sweep(sky)
    cand = [r for r in sweep if r["k"] >= 1]
    best = min(cand, key=lambda r: (abs(r["k"] - target_k), -r["clustered"], r["mu"], r["eps"]))
    return best["eps"], best["mu"], cluster_sky_network(sky, best["eps"], best["mu"])


def partition(sky: SkyNetwork, target_k: int) -> ClusterResult:
    if target_k <= 1:
        return single_cluster(sky)
    return calibrate_for_target_k(sky, target_k)[2]


def build_hypergraph(sky: SkyNetwork, result: ClusterResult) -> nx.Graph:
    """Sky network hyper-graph G^f over gateway, bridge and outlier stations.

    Real edges: segments leaving a cluster. Virtual intra-cluster edges join
    every pair of gateways of the same cluster (weight from the straight-line
    distance), giving DaaS-HGC its abstract view of each region.
    """
    hg = nx.Graph()
    special = sorted(result.gateway | result.bridge | result.outlier)
    special_set = set(special)
    for rs in special:
        hg.add_node(rs, cluster=result.cluster_of(rs))
    for u, v, data in sky.graph.edges(data=True):
        if u in special_set and v in special_set:
            cu, cv = result.cluster_of(u), result.cluster_of(v)
            if cu is None or cv is None or cu != cv:
                hg.add_edge(u, v, weight=_edge_weight(sky, u, v, data["distance"]), virtual=False)
    by_cluster: dict[int, list[str]] = {}
    for rs in special:
        c = result.cluster_of(rs)
        if c is not None:
            by_cluster.setdefault(c, []).append(rs)
    for c, gws in by_cluster.items():
        for i, u in enumerate(gws):
            for v in gws[i + 1:]:
                if not hg.has_edge(u, v):
                    hg.add_edge(u, v, weight=_edge_weight(sky, u, v, sky.euclid(u, v)),
                                virtual=True, cluster=c)
    return hg


def _edge_weight(sky: SkyNetwork, u: str, v: str, dist: float) -> float:
    return dist + 0.5 * (sky.waiting_time(u) + sky.charging_cost(u) + sky.waiting_time(v) + sky.charging_cost(v))
