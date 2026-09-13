"""DaaS-SG: single-graph baseline (no clustering).

Treats the whole sky network as one unified graph and runs an exact A*
search from source to target, using the same path scoring function
(Eq. "eq:path-score") and the same drone-formation selection (Algorithm 3,
Eq. "eq:daas-score") as DaaS-GC/DaaS-HGC. 
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.composition import Formation, select_drone_formation
from daasgc.path_search import PathResult, a_star_search
from daasgc.sky_network import SkyNetwork


@dataclass
class DaaSResult:
    path: PathResult
    formation: Formation | None


def run_daas_sg(sky: SkyNetwork, source: str, target: str, items: list[dict],
                weights: dict[str, float]) -> DaaSResult:
    path = a_star_search(sky, source, target, None)
    if path is None:
        raise ValueError(f"No path found between {source} and {target} in the sky network.")
    formation = select_drone_formation(sky, path.stations, items, weights)
    return DaaSResult(path=path, formation=formation)
