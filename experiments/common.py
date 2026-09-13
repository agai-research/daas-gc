"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import os
import time

from daasgc.runner import run_daas_gc, run_daas_hgc
from daasgc.sky_network import SkyNetwork
from baselines.daas_kge.run import run_daas_kge
from baselines.daas_sg import run_daas_sg

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "instances")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def instance_dir(n_stations: int, n_drones: int) -> str:
    return os.path.join(DATA_DIR, f"s{n_stations}_d{n_drones}")


def load_instance(n_stations: int, n_drones: int):
    d = instance_dir(n_stations, n_drones)
    sky = SkyNetwork.from_json(os.path.join(d, "stations.json"),
                                os.path.join(d, "segments.json"),
                                os.path.join(d, "drones.json"))
    with open(os.path.join(d, "requests.json")) as f:
        requests = json.load(f)
    with open(os.path.join(d, "daas_catalog.json")) as f:
        catalog = json.load(f)
    return sky, requests, catalog


def run_all_methods(sky, catalog, request: dict, target_k: int | None = 5,
                     kge_walks: int = 3, kge_walk_len: int = 3, kge_dim: int = 32) -> dict:
    """Run DaaS-GC, DaaS-HGC, DaaS-SG and DaaS-KGE on one request, timing
    each and returning a uniform result dict per method.
    """
    src, tgt, items, weights = request["source"], request["target"], request["items"], request["weights"]
    out = {}

    t0 = time.perf_counter()
    r = run_daas_gc(sky, src, tgt, items, weights, target_k=target_k)
    dt = time.perf_counter() - t0
    out["DaaS-GC"] = _to_record(sky, r.path, r.formation, dt, k=r.cluster_result.k)

    t0 = time.perf_counter()
    r = run_daas_hgc(sky, src, tgt, items, weights, target_k=target_k)
    dt = time.perf_counter() - t0
    out["DaaS-HGC"] = _to_record(sky, r.path, r.formation, dt, k=r.cluster_result.k)

    t0 = time.perf_counter()
    r = run_daas_sg(sky, src, tgt, items, weights)
    dt = time.perf_counter() - t0
    out["DaaS-SG"] = _to_record(sky, r.path, r.formation, dt, k=1)

    t0 = time.perf_counter()
    r = run_daas_kge(sky, catalog, src, tgt, items, weights,
                      num_walks=kge_walks, walk_length=kge_walk_len, dimension=kge_dim)
    dt = time.perf_counter() - t0
    out["DaaS-KGE"] = _to_record(sky, r.path, r.formation, dt, k=None)

    return out


def _to_record(sky, path, formation, dt: float, k) -> dict:
    stations = path.stations
    distance = sum(sky.segment_distance(stations[i], stations[i + 1]) for i in range(len(stations) - 1)) \
        if len(stations) > 1 else 0.0
    n_drones = len(formation.drones) if formation is not None else 0

    delivery_cost, delivery_time, failure_rate = 0.0, 0.0, 1.0
    if formation is not None:
        drones = formation.drones
        delivery_cost = sum(d.cost_per_km * distance for d in drones)
        per_drone_time = [distance / max(d.speed_kmh, 1e-6)
                          + sum(sky.waiting_time(s) + sky.charging_cost(s) / max(d.speed_kmh, 1e-6)
                                for s in stations)
                          for d in drones]
        delivery_time = max(per_drone_time) if per_drone_time else 0.0
        # Failure-rate proxy: grows with formation size and with the ratio
        # of requested payload to available payload margin (more drones
        # and tighter margins => higher chance of an SLA violation).
        avg_availability = sum(d.availability for d in drones) / len(drones)
        failure_rate = round(min(1.0, (1 - avg_availability) + 0.03 * (n_drones - 1)), 4)

    return {
        "path_stations": stations,
        "path_score": path.score,
        "n_stations": len(stations),
        "distance": round(distance, 3),
        "n_drones": n_drones,
        "formation_score": None if formation is None else formation.score,
        "delivery_cost": round(delivery_cost, 3),
        "delivery_time_min": round(delivery_time, 3),
        "failure_rate": failure_rate,
        "computation_time_s": dt,
        "k": k,
        "failed": formation is None,
    }
