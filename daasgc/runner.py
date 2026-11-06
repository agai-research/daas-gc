"""DaaS-GC / DaaS-HGC pipelines, split into pre-processing and query.

Pre-processing (once per network and k): (eps, mu) calibration, clustering,
regions and hyper-graph -> `prepare(sky, k)`.
Query (per delivery request, the part that is timed in the experiments):
path search + drone formation selection -> `query_daas_gc`, `query_daas_hgc`.
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.clustering import partition
from daasgc.composition import Formation, select_drone_formation
from daasgc.hypergraph import hgc_path_search
from daasgc.path_search import Partition, PathResult, delivery_path_search, prepare_partition
from daasgc.sky_network import SkyNetwork


@dataclass
class DaaSResult:
    path: PathResult
    formation: Formation | None


def prepare(sky: SkyNetwork, target_k: int) -> Partition:
    key = ("partition", target_k)
    if key not in sky._cache:
        sky._cache[key] = prepare_partition(sky, partition(sky, target_k))
    return sky._cache[key]


def query_daas_gc(sky, part: Partition, source, target, items, weights) -> DaaSResult:
    path = delivery_path_search(sky, source, target, part)
    return DaaSResult(path, select_drone_formation(sky, path.stations, items, weights))


def query_daas_hgc(sky, part: Partition, source, target, items, weights) -> DaaSResult:
    path = hgc_path_search(sky, source, target, part)
    return DaaSResult(path, select_drone_formation(sky, path.stations, items, weights))


def run_daas_gc(sky: SkyNetwork, source, target, items, weights, target_k: int = 5) -> DaaSResult:
    return query_daas_gc(sky, prepare(sky, target_k), source, target, items, weights)


def run_daas_hgc(sky: SkyNetwork, source, target, items, weights, target_k: int = 5) -> DaaSResult:
    return query_daas_hgc(sky, prepare(sky, target_k), source, target, items, weights)
