"""Sky network model for DaaS-GC.

A sky network is a weighted graph of recharging stations (nodes, with pad
and charging attributes and planar coordinates in km) connected by skyway
segments (edges, with a flight distance in km). The drone fleet is loaded
alongside, so clustering, path search and composition share one object.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import networkx as nx

FULL_RECHARGE_MIN = 60.0
# Charging-speed factor of a station's charging voltage (higher voltage = faster).
VOLTAGE_TIME_FACTOR = {12: 1.25, 24: 1.0, 48: 0.8}


@dataclass
class Drone:
    """A drone service (DaaS)."""

    id: str
    payload: float            # max payload (kg)
    empty_mass: float         # kg
    battery: float            # battery capacity (Wh)
    wh_per_km: float          # consumption when empty (Wh/km)
    range_km: float           # empty range (km)
    speed_kmh: float
    cost_per_km: float
    availability: float       # 0..1
    reputation: float         # 0..1
    supported_pads: list[str] = field(default_factory=list)
    home_station: str = ""
    service_price: float = 0.0  # price of the drone's DaaS offer (cheapest in the catalog)

    def rate(self, load_kg: float) -> float:
        """Energy per km (Wh/km) when carrying `load_kg`: linear in total mass."""
        return self.wh_per_km * (self.empty_mass + load_kg) / self.empty_mass


class SkyNetwork:
    def __init__(self, graph: nx.Graph, drones: list[Drone]):
        self.graph = graph
        self.drones = drones
        self._cache: dict = {}

    @classmethod
    def from_dir(cls, instance_dir: str) -> "SkyNetwork":
        def load(name):
            with open(os.path.join(instance_dir, name)) as f:
                return json.load(f)
        sky = cls.from_records(load("stations.json"), load("segments.json"), load("drones.json"))
        cat = os.path.join(instance_dir, "daas_catalog.json")
        if os.path.exists(cat):
            sky.attach_catalog(load("daas_catalog.json"))
        return sky

    def attach_catalog(self, catalog: list[dict]) -> None:
        """Set each drone's service price to its cheapest DaaS offer in the catalog."""
        best: dict[str, float] = {}
        for row in catalog:
            best[row["drone_id"]] = min(row["price"], best.get(row["drone_id"], float("inf")))
        for d in self.drones:
            d.service_price = best.get(d.id, 0.0)

    @classmethod
    def from_json(cls, stations_path: str, segments_path: str, drones_path: str) -> "SkyNetwork":
        with open(stations_path) as f:
            stations = json.load(f)
        with open(segments_path) as f:
            segments = json.load(f)
        with open(drones_path) as f:
            drones = json.load(f)
        return cls.from_records(stations, segments, drones)

    @classmethod
    def from_records(cls, stations, segments, drones) -> "SkyNetwork":
        g = nx.Graph()
        for st in stations:
            g.add_node(st["id"], **st)
        for seg in segments:
            g.add_edge(seg["source"], seg["target"], distance=seg["distance"])
        return cls(g, [Drone(**d) for d in drones])

    # ---- accessors -------------------------------------------------------
    def station(self, sid: str) -> dict:
        return self.graph.nodes[sid]

    def neighbors(self, sid: str) -> set[str]:
        """Structural neighbourhood N(rs) = neighbours(rs) U {rs}."""
        return set(self.graph.neighbors(sid)) | {sid}

    def segment_distance(self, u: str, v: str) -> float:
        return self.graph.edges[u, v]["distance"]

    def euclid(self, u: str, v: str) -> float:
        a, b = self.graph.nodes[u], self.graph.nodes[v]
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    def waiting_time(self, sid: str) -> float:
        return self.graph.nodes[sid].get("waiting_time", 0.0)

    def charging_cost(self, sid: str) -> float:
        return self.graph.nodes[sid].get("charging_cost", 0.0)

    def functional_profile(self, sid: str) -> tuple:
        """Charging characteristics that decide drone compatibility (pad type)."""
        return (self.graph.nodes[sid]["pad_type"],)

    def compatible(self, drone: Drone, sid: str) -> bool:
        return self.graph.nodes[sid]["pad_type"] in drone.supported_pads

    def charge_minutes(self, drone: Drone, sid: str, wh: float) -> float:
        v = self.graph.nodes[sid]["charging_voltage"]
        return FULL_RECHARGE_MIN * (wh / drone.battery) * VOLTAGE_TIME_FACTOR.get(v, 1.0)

    def free_pads(self, sid: str) -> int:
        st = self.graph.nodes[sid]
        return max(1, st["n_pads"] - round(st["congestion"] * st["n_pads"]))

    def path_distance(self, stations: list[str]) -> float:
        return sum(self.segment_distance(stations[i], stations[i + 1]) for i in range(len(stations) - 1))

    def n_stations(self) -> int:
        return self.graph.number_of_nodes()

    def n_segments(self) -> int:
        return self.graph.number_of_edges()
