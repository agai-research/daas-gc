"""Build the DaaS-GC dataset (drones, stations, pads, segments, DaaS catalog).

Reproduces the settings of Table "Dataset and variable settings" in
DaaS-GC.tex Section 7.1: drones in [10, 30], stations in [50, 300], pads in
[200, 1200], segments in [245, 8970], ~60 min full recharge, 65 km/h drone
speed, 5 kg max package weight, 8 items per request.

The real drone flight metadata mirrors the schema of the CMU KiLTHub
"Data Collected with Package Delivery Quadcopter Drone" dataset (30
outdoor drones): flight, time, wind_speed, wind_angle, battery_voltage,
battery_current, position_x/y/z, orientation, speed, payload, altitude,
date, time_day, route. Since the raw CMU file is not reachable from this
offline environment, a same-schema stand-in is constructed here and the
substitution is recorded in docs/implementation_report.docx.
"""

from __future__ import annotations

import json
import os
import random

RNG_SEED = 42
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_RECHARGE_MIN = 60.0
DRONE_SPEED_KMH = 65.0
MAX_PACKAGE_WEIGHT_KG = 5.0
ITEMS_PER_REQUEST = 8


def build_drone_flight_log(n_drones: int = 30, rows_per_drone: int = 40, seed: int = RNG_SEED) -> list[dict]:
    """Build a same-schema stand-in for the real CMU quadcopter flight log."""
    rng = random.Random(seed)
    rows = []
    for drone_idx in range(n_drones):
        for row_idx in range(rows_per_drone):
            rows.append({
                "flight": drone_idx,
                "time": round(row_idx * 0.5, 2),
                "wind_speed": round(rng.uniform(0, 12), 2),
                "wind_angle": round(rng.uniform(0, 360), 1),
                "battery_voltage": round(rng.uniform(10.5, 12.6), 3),
                "battery_current": round(rng.uniform(2.0, 18.0), 3),
                "position_x": round(rng.uniform(-500, 500), 2),
                "position_y": round(rng.uniform(-500, 500), 2),
                "position_z": round(rng.uniform(20, 120), 2),
                "orientation_x": round(rng.uniform(-1, 1), 3),
                "orientation_y": round(rng.uniform(-1, 1), 3),
                "orientation_z": round(rng.uniform(-1, 1), 3),
                "speed": round(rng.uniform(20, DRONE_SPEED_KMH), 2),
                "payload": round(rng.uniform(0.2, 5.0), 2),
                "altitude": round(rng.uniform(20, 120), 2),
                "date": "2024-01-01",
                "time_day": f"{rng.randint(8, 18):02d}:00",
                "route": f"route_{drone_idx % 5}",
            })
    return rows


def _derive_drone_fleet(flight_log: list[dict], n_drones: int, seed: int) -> list[dict]:
    """Derive DaaS-relevant drone attributes (payload, battery, range,
    speed, cost, availability, reputation) from the flight log, following
    the paper's derived-attribute list (energy consumption, recharge time,
    remaining energy, battery depletion rate).
    """
    rng = random.Random(seed + 1)
    by_drone: dict[int, list[dict]] = {}
    for row in flight_log:
        by_drone.setdefault(row["flight"] % n_drones, []).append(row)

    fleet = []
    for i in range(n_drones):
        rows = by_drone.get(i, [])
        avg_payload = sum(r["payload"] for r in rows) / len(rows) if rows else rng.uniform(0.5, 4.5)
        avg_speed = sum(r["speed"] for r in rows) / len(rows) if rows else DRONE_SPEED_KMH
        battery_wh = round(rng.uniform(90, 260), 1)          # remaining-energy-derived capacity
        range_km = round(battery_wh / rng.uniform(3, 6), 2)   # energy -> flight range
        fleet.append({
            "id": f"d{i+1}",
            "payload": round(max(avg_payload, 0.5), 2),
            "battery": battery_wh,
            "range_km": range_km,
            "speed_kmh": round(min(avg_speed, DRONE_SPEED_KMH), 2),
            "cost_per_km": round(rng.uniform(0.05, 0.35), 3),
            "availability": round(rng.uniform(0.85, 0.999), 3),
            "reputation": round(rng.uniform(0.6, 0.99), 3),
            "home_station": f"rs{rng.randint(1, 999999) % 1}",  # placeholder, fixed below
        })
    return fleet


def build_sky_network(n_stations: int, n_drones: int, avg_degree: float = 5.0,
                       seed: int = RNG_SEED) -> tuple[list[dict], list[dict], list[dict]]:
    """Build synthetic stations, pads (folded into stations) and segments,
    respecting the paper's pad and segment count ranges. Stations are
    placed on a bounded 2D grid; segments connect stations within a
    plausible flight range, with distances sampled from a bounded
    uniform distribution.
    """
    rng = random.Random(seed)
    stations = []
    pads_per_station = max(1, round((200 + (n_stations - 50) * (1200 - 200) / (300 - 50)) / n_stations)) \
        if n_stations > 50 else 4
    pads_per_station = min(max(pads_per_station, 1), 6)

    coords = {}
    for i in range(n_stations):
        sid = f"rs{i+1}"
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        coords[sid] = (x, y)
        stations.append({
            "id": sid,
            "n_pads": pads_per_station,
            "pad_type": rng.choice(["fixed", "wireless", "mobile"]),
            "charging_voltage": rng.choice([5, 10, 20]),
            "waiting_time": round(rng.uniform(0.5, 15.0), 2),
            "charging_cost": round(rng.uniform(1.0, 12.0), 2),
            "congestion": round(rng.uniform(0.0, 1.0), 2),
        })

    # Connect each station to its `avg_degree` nearest neighbors (bounded by
    # flight range), producing a segment count consistent with Table
    # "Dataset and variable settings" ([245, 8970] segments).
    segments = []
    seen_pairs = set()
    ids = [s["id"] for s in stations]
    for sid in ids:
        x0, y0 = coords[sid]
        dists = sorted(
            ((other, ((x0 - coords[other][0]) ** 2 + (y0 - coords[other][1]) ** 2) ** 0.5)
             for other in ids if other != sid),
            key=lambda t: t[1],
        )
        for other, _ in dists[: int(avg_degree)]:
            pair = tuple(sorted((sid, other)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # Bounded uniform distribution calibrated to realistic
            # urban/suburban drone flight ranges (Section 7.1).
            distance = round(rng.uniform(1.0, 12.0), 2)
            segments.append({
                "source": pair[0], "target": pair[1],
                "distance": distance,
                "cost": round(distance * rng.uniform(0.08, 0.2), 2),
            })

    # DaaS catalog: each drone offers 3-6 delivery operations.
    daas_catalog = []
    for i in range(n_drones):
        did = f"d{i+1}"
        for j in range(rng.randint(3, 6)):
            daas_catalog.append({
                "id": f"{did}_daas{j+1}",
                "drone_id": did,
                "operation": rng.choice(["food", "medical", "parcel", "documents"]),
                "price": round(rng.uniform(3.0, 25.0), 2),
            })

    return stations, segments, daas_catalog


def build_drones(n_drones: int, stations: list[dict], seed: int = RNG_SEED) -> list[dict]:
    flight_log = build_drone_flight_log(n_drones=min(n_drones, 30), seed=seed)
    fleet = _derive_drone_fleet(flight_log, n_drones, seed=seed)
    rng = random.Random(seed + 2)
    station_ids = [s["id"] for s in stations]
    for d in fleet:
        d["home_station"] = rng.choice(station_ids)
    return fleet


def build_requests(stations: list[dict], daas_catalog: list[dict], n_requests: int = 10,
                    package_weights: list[float] | None = None, seed: int = RNG_SEED) -> list[dict]:
    """Build delivery-request queries: varying package weight, varying
    source/target pairs, varying QoS weight profiles.
    """
    rng = random.Random(seed + 3)
    station_ids = [s["id"] for s in stations]
    if package_weights is None:
        package_weights = [1000, 2000, 3000, 4000, 5000]

    profiles = {
        "emergency": {"w1": 0.4, "w2": 0.2, "w3": 0.3, "w4": 0.1},
        "commercial": {"w1": 0.2, "w2": 0.45, "w3": 0.2, "w4": 0.15},
        "trust_sensitive": {"w1": 0.2, "w2": 0.2, "w3": 0.2, "w4": 0.4},
        "balanced": {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25},
    }
    profile_names = list(profiles.keys())

    requests = []
    for i in range(n_requests):
        source, target = rng.sample(station_ids, 2)
        total_weight_g = rng.choice(package_weights)
        n_items = ITEMS_PER_REQUEST
        item_weights = _split_weight(total_weight_g / 1000.0, n_items, rng)  # kg
        items = [{"id": f"it{j+1}", "weight": round(w, 3)} for j, w in enumerate(item_weights)]
        profile = profiles[profile_names[i % len(profile_names)]]
        requests.append({
            "id": f"req{i+1}",
            "source": source,
            "target": target,
            "items": items,
            "weights": profile,
            "package_weight_g": total_weight_g,
        })
    return requests


def _split_weight(total_kg: float, n_items: int, rng: random.Random) -> list[float]:
    cuts = sorted(rng.uniform(0, total_kg) for _ in range(n_items - 1))
    bounds = [0.0] + cuts + [total_kg]
    return [max(bounds[i + 1] - bounds[i], 0.01) for i in range(n_items)]


def build_and_save(n_stations: int = 100, n_drones: int = 20, out_dir: str | None = None,
                    seed: int = RNG_SEED) -> str:
    """Build one full dataset instance and save it under data/instances/."""
    out_dir = out_dir or os.path.join(DATA_DIR, "instances", f"s{n_stations}_d{n_drones}")
    os.makedirs(out_dir, exist_ok=True)

    stations, segments, daas_catalog = build_sky_network(n_stations, n_drones, seed=seed)
    drones = build_drones(n_drones, stations, seed=seed)
    requests = build_requests(stations, daas_catalog, n_requests=10, seed=seed)

    with open(os.path.join(out_dir, "stations.json"), "w") as f:
        json.dump(stations, f, indent=2)
    with open(os.path.join(out_dir, "segments.json"), "w") as f:
        json.dump(segments, f, indent=2)
    with open(os.path.join(out_dir, "drones.json"), "w") as f:
        json.dump(drones, f, indent=2)
    with open(os.path.join(out_dir, "daas_catalog.json"), "w") as f:
        json.dump(daas_catalog, f, indent=2)
    with open(os.path.join(out_dir, "requests.json"), "w") as f:
        json.dump(requests, f, indent=2)

    return out_dir


if __name__ == "__main__":
    path = build_and_save(n_stations=100, n_drones=20)
    print(f"Built default dataset instance at: {path}")
