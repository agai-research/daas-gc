"""End-to-end DaaS-KGE pipeline.

1. Build the DaaSKG (kg_build.py).
2. Guided random walks + Skip-Gram embedding (embedding.py).
3. Extract high-proximity candidate stations/drones (proximity.py).
4. Proximity-aware drone formation selection (composition.py).
5. Skyway path selection over the candidate subspace (path_selection.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.sky_network import SkyNetwork
from baselines.daas_kge.composition import KGEFormation, compose_kge_formation
from baselines.daas_kge.embedding import generate_walks, train_embeddings
from baselines.daas_kge.kg_build import build_daaskg
from baselines.daas_kge.path_selection import KGEPathResult, select_skyway_path
from baselines.daas_kge.proximity import candidate_drones, candidate_stations


@dataclass
class KGEResult:
    path: KGEPathResult
    formation: KGEFormation | None


def run_daas_kge(sky: SkyNetwork, daas_catalog: list[dict], source: str, target: str,
                  items: list[dict], weights: dict[str, float],
                  num_walks: int = 5, walk_length: int = 4, window: int = 3,
                  dimension: int = 64) -> KGEResult:
    kg = build_daaskg(sky, daas_catalog)
    walks = generate_walks(kg, num_walks=num_walks, walk_length=walk_length)
    model = train_embeddings(walks, dimension=dimension, window=window)

    station_ids = [n["id"] for n in [{"id": s} for s in sky.graph.nodes]]
    cand_stations = candidate_stations(model, source, target, station_ids, top_n=max(20, len(station_ids) // 4))
    cand_drone_ids = candidate_drones(sky, model, cand_stations, top_n=max(15, len(sky.drones) // 3))

    path = select_skyway_path(sky, source, target, cand_stations)
    if path is None:
        raise ValueError(f"DaaS-KGE could not find a path between {source} and {target}.")

    max_segment = max((sky.segment_distance(path.stations[i], path.stations[i + 1])
                        for i in range(len(path.stations) - 1)), default=0.0)
    formation = compose_kge_formation(sky, cand_drone_ids, items, weights, max_segment)

    return KGEResult(path=path, formation=formation)
