"""Sky network model for DaaS-GC.

A sky network is a weighted graph of recharging stations (nodes) connected by
skyway segments (edges). Stations carry pad/charging attributes; segments
carry a flight distance. Drone services and delivery items are loaded
alongside the graph so the rest of the pipeline (clustering, path search,
composition) can access them from one place.

Maps directly to Table "Basic symbols and notations" in DaaS-GC.tex.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Drone:
    """A single drone service (DaaS)."""

    id: str
    payload: float          # kg
    battery: float           # mAh
    range_km: float          # max flight range
    speed_kmh: float
    cost_per_km: float
    availability: float       # 0..1
    reputation: float         # 0..1
    home_station: str


@dataclass
class DeliveryRequest:
    """A user delivery request. See Definition 1 in DaaS-GC.tex."""

    id: str
    source: str
    target: str
    items: list[dict[str, float]]   # [{"id": .., "weight": ..}, ...]
    weights: dict[str, float] = field(default_factory=lambda: {
        "w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25,
    })

    @property
    def total_weight(self) -> float:
        return sum(it["weight"] for it in self.items)


class SkyNetwork:
    """Wraps a NetworkX graph plus the drone fleet for a delivery scenario."""

    def __init__(self, graph: nx.Graph, drones: list[Drone]):
        self.graph = graph
        self.drones = drones

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, stations_path: str, segments_path: str, drones_path: str) -> "SkyNetwork":
        with open(stations_path) as f:
            stations = json.load(f)
        with open(segments_path) as f:
            segments = json.load(f)
        with open(drones_path) as f:
            drones_raw = json.load(f)

        g = nx.Graph()
        for st in stations:
            g.add_node(st["id"], **st)
        for seg in segments:
            g.add_edge(seg["source"], seg["target"], distance=seg["distance"],
                       cost=seg.get("cost", seg["distance"] * 0.1))

        drones = [Drone(**d) for d in drones_raw]
        return cls(g, drones)

    def to_graphml(self, path: str) -> None:
        nx.write_graphml(self.graph, path)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def station(self, station_id: str) -> dict[str, Any]:
        return self.graph.nodes[station_id]

    def neighbors(self, station_id: str) -> set[str]:
        """Structural neighborhood N(rs) = neighbors(rs) U {rs} (Def. structural similarity)."""
        return set(self.graph.neighbors(station_id)) | {station_id}

    def segment_distance(self, u: str, v: str) -> float:
        return self.graph.edges[u, v]["distance"]

    def waiting_time(self, station_id: str) -> float:
        return self.graph.nodes[station_id].get("waiting_time", 0.0)

    def charging_cost(self, station_id: str) -> float:
        return self.graph.nodes[station_id].get("charging_cost", 0.0)

    def n_stations(self) -> int:
        return self.graph.number_of_nodes()

    def n_segments(self) -> int:
        return self.graph.number_of_edges()
