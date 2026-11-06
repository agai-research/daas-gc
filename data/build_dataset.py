"""Build DaaS-GC dataset instances (sky network + drone fleet + DaaS catalog).

* stations in [50, 300]; skyway segments = the SEGMENT_DENSITY share of the
  closest station pairs, i.e. round(0.2 * n(n-1)/2) segments
  -> 245 segments for 50 stations ... 8 970 for 300 stations;
* ~4 charging pads per station on average -> ~[200, 1200] pads;
* full recharging time ~60 min, drone speed 65 km/h, max package 5 kg,
  8 items per request (the last two are used by the request generator in
  experiments/common.py).

Stations are placed uniformly in a REGION_KM x REGION_KM airspace (uniform
prior, Section "Dataset and experimental setup"); a segment's distance is the
Euclidean distance between its two stations, so segments only join stations
that are within flight range of each other.

Drone attributes: the raw CMU KiLTHub quadcopter log is not redistributed in
this repository, so a same-schema synthetic stand-in fleet is generated with
the distributions documented below (see README, "Data provenance").

Seeding (one seed per artefact, no collisions):
  network of n stations -> NETWORK_SEED_BASE + n   (shared by all fleet sizes)
  drone fleet           -> FLEET_SEED              (the first d drones of one
                                                    master fleet of 30 drones)
"""

from __future__ import annotations

import json
import math
import os
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(DATA_DIR, "instances")

REGION_KM = 100.0
SEGMENT_DENSITY = 0.2
FULL_RECHARGE_MIN = 60.0
DRONE_SPEED_KMH = 65.0
MAX_PACKAGE_WEIGHT_KG = 5.0
ITEMS_PER_REQUEST = 8
MAX_FLEET = 30

NETWORK_SEED_BASE = 10_000
FLEET_SEED = 20_000

PAD_TYPES = ["fixed", "wireless", "mobile"]
VOLTAGES = [12, 24, 48]


def network_seed(n_stations: int) -> int:
    return NETWORK_SEED_BASE + n_stations


def build_network(n_stations: int, seed: int | None = None) -> tuple[list[dict], list[dict]]:
    rng = random.Random(network_seed(n_stations) if seed is None else seed)
    stations, xy = [], []
    for i in range(n_stations):
        x, y = rng.uniform(0, REGION_KM), rng.uniform(0, REGION_KM)
        xy.append((x, y))
        stations.append({
            "id": f"rs{i + 1}",
            "x": round(x, 3), "y": round(y, 3),
            "n_pads": rng.randint(2, 6),                      # mean 4 pads/station
            "pad_type": rng.choice(PAD_TYPES),
            "charging_voltage": rng.choice(VOLTAGES),
            "waiting_time": round(rng.uniform(1.0, 15.0), 2),  # average waiting time (min)
            "charging_cost": round(rng.uniform(1.0, 12.0), 2),  # fee per charging session
            "congestion": round(rng.uniform(0.0, 0.8), 2),     # share of pads usually busy
        })

    pairs = sorted(
        (math.dist(xy[i], xy[j]), i, j) for i in range(n_stations) for j in range(i + 1, n_stations)
    )
    m = round(SEGMENT_DENSITY * n_stations * (n_stations - 1) / 2)
    chosen = pairs[:m]

    # Guarantee a connected network (union-find over the remaining pairs).
    parent = list(range(n_stations))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, i, j in chosen:
        parent[find(i)] = find(j)
    for d, i, j in pairs[m:]:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            chosen.append((d, i, j))

    segments = [{
        "source": stations[i]["id"], "target": stations[j]["id"],
        "distance": round(d, 3),
    } for d, i, j in chosen]
    return stations, segments


def build_fleet(n_stations: int, n_drones: int, seed: int = FLEET_SEED) -> tuple[list[dict], list[dict]]:
    if not 1 <= n_drones <= MAX_FLEET:
        raise ValueError(f"n_drones must be in [1, {MAX_FLEET}]")
    rng = random.Random(seed)
    master, catalog = [], []
    for i in range(MAX_FLEET):
        empty_mass = round(rng.uniform(2.0, 6.0), 2)          # kg
        battery = round(rng.uniform(200.0, 500.0), 1)         # Wh
        wh_per_km = round(rng.uniform(4.0, 7.0), 3)           # Wh/km, empty
        home_u = rng.random()
        master.append({
            "id": f"d{i + 1}",
            "payload": round(rng.uniform(0.8, 3.0), 2),       # max payload (kg)
            "empty_mass": empty_mass,
            "battery": battery,
            "wh_per_km": wh_per_km,
            "range_km": round(battery / wh_per_km, 2),        # empty range
            "speed_kmh": DRONE_SPEED_KMH,
            "cost_per_km": round(rng.uniform(0.3, 1.2), 3),
            "availability": round(rng.uniform(0.85, 0.99), 3),
            "reputation": round(rng.uniform(0.6, 0.99), 3),
            "supported_pads": sorted(rng.sample(PAD_TYPES, 2)),
            "home_station": f"rs{min(n_stations, int(home_u * n_stations) + 1)}",
        })
        for j in range(rng.randint(3, 6)):
            catalog.append({
                "id": f"d{i + 1}_daas{j + 1}", "drone_id": f"d{i + 1}",
                "operation": rng.choice(["food", "medical", "parcel", "documents"]),
                "price": round(rng.uniform(3.0, 25.0), 2),
            })
    keep = {d["id"] for d in master[:n_drones]}
    return master[:n_drones], [c for c in catalog if c["drone_id"] in keep]


def instance_dir(n_stations: int, n_drones: int) -> str:
    return os.path.join(INSTANCES_DIR, f"s{n_stations}_d{n_drones}")


def build_and_save(n_stations: int = 100, n_drones: int = 20, out_dir: str | None = None) -> str:
    out_dir = out_dir or instance_dir(n_stations, n_drones)
    os.makedirs(out_dir, exist_ok=True)
    stations, segments = build_network(n_stations)
    drones, catalog = build_fleet(n_stations, n_drones)
    for name, obj in (("stations", stations), ("segments", segments),
                      ("drones", drones), ("daas_catalog", catalog)):
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=1)
    return out_dir


def ensure_instance(n_stations: int, n_drones: int) -> str:
    d = instance_dir(n_stations, n_drones)
    needed = ["stations.json", "segments.json", "drones.json", "daas_catalog.json"]
    if not all(os.path.exists(os.path.join(d, f)) for f in needed):
        build_and_save(n_stations, n_drones)
    return d


if __name__ == "__main__":
    print(build_and_save(100, 20))
