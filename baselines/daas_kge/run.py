"""DaaS-KGE pipeline, split into pre-processing and query.

Pre-processing (once per network/fleet): build the DaaSKG, guided random
walks, Skip-Gram training -> `prepare_kge`.
Query (timed): candidate stations/drones from the embedding subspace, path
selection over the candidate subspace, proximity-aware composition.
"""

from __future__ import annotations

from dataclasses import dataclass

from baselines.daas_kge.composition import compose_kge_formation
from baselines.daas_kge.embedding import generate_walks, train_embeddings
from baselines.daas_kge.kg_build import build_daaskg
from baselines.daas_kge.path_selection import select_skyway_path
from baselines.daas_kge.proximity import candidate_drones, candidate_stations
from daasgc.runner import DaaSResult
from daasgc.path_search import PathResult
from daasgc.sky_network import SkyNetwork


@dataclass
class KGEModel:
    model: object


def prepare_kge(sky: SkyNetwork, daas_catalog: list[dict], num_walks: int = 3, walk_length: int = 4,
                window: int = 3, dimension: int = 32, seed: int = 42) -> KGEModel:
    key = ("kge", num_walks, walk_length, window, dimension, seed)
    if key not in sky._cache:
        kg = build_daaskg(sky, daas_catalog)
        walks = generate_walks(kg, num_walks=num_walks, walk_length=walk_length, seed=seed)
        sky._cache[key] = KGEModel(train_embeddings(walks, dimension=dimension, window=window, seed=seed))
    return sky._cache[key]


def query_daas_kge(sky: SkyNetwork, kge: KGEModel, source, target, items, weights) -> DaaSResult:
    station_ids = list(sky.graph.nodes)
    cand_st = candidate_stations(kge.model, source, target, station_ids, top_n=max(20, len(station_ids) // 4))
    cand_dr = candidate_drones(sky, kge.model, cand_st, top_n=max(15, len(sky.drones) // 3))
    kp = select_skyway_path(sky, source, target, cand_st)
    if kp is None:
        raise ValueError(f"DaaS-KGE found no path between {source} and {target}")
    path = PathResult(kp.stations, kp.score, "embedding-subspace")
    return DaaSResult(path, compose_kge_formation(sky, kp.stations, cand_dr, items, weights))


def run_daas_kge(sky: SkyNetwork, daas_catalog, source, target, items, weights, **kw) -> DaaSResult:
    return query_daas_kge(sky, prepare_kge(sky, daas_catalog, **kw), source, target, items, weights)
