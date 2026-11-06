"""DaaS-SG: single-graph baseline (no clustering).

Exact A* over the whole, unpartitioned sky network with the same edge cost,
heuristic and path score as DaaS-GC, followed by the same drone formation
selection (Algorithm 3). DaaS-GC/DaaS-HGC with k = 1 run exactly this search.
"""

from __future__ import annotations

from daasgc.composition import select_drone_formation
from daasgc.path_search import a_star_search
from daasgc.runner import DaaSResult
from daasgc.sky_network import SkyNetwork


def run_daas_sg(sky: SkyNetwork, source, target, items, weights) -> DaaSResult:
    path = a_star_search(sky, source, target, None)
    if path is None:
        raise ValueError(f"No path between {source} and {target}")
    path.scenario = "single-graph"
    return DaaSResult(path, select_drone_formation(sky, path.stations, items, weights))
