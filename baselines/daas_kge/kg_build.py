"""Build the DaaS knowledge graph (DaaSKG), Definition 6 of DaaS_KGE-1.pdf.

DaaSKG = (E, R, D+): typed entities (drones Ed, delivery services Es,
stations Er, pads Ep) linked by typed relations: ATTEND (drone->station),
PROVIDE (drone->DaaS), SkySegment (station->station), BELONGTO
(pad->station), plus similarity edges D-SIMILAR / S-SIMILAR / R-SIMILAR
built from feature closeness (used to seed the metapath2vec random walks).
"""

from __future__ import annotations

import itertools

import networkx as nx

from daasgc.sky_network import SkyNetwork


def _feature_close(a: dict, b: dict, keys: list[str], tol: float = 0.25) -> bool:
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            denom = max(abs(va), abs(vb), 1e-6)
            if abs(va - vb) / denom > tol:
                return False
        elif va != vb:
            return False
    return True


def build_daaskg(sky: SkyNetwork, daas_catalog: list[dict]) -> nx.MultiDiGraph:
    """Build the typed multi-relational DaaSKG."""
    kg = nx.MultiDiGraph()

    for st_id, st_data in sky.graph.nodes(data=True):
        node = f"S::{st_id}"
        kg.add_node(node, kind="station", **st_data)
        for p in range(st_data.get("n_pads", 1)):
            pad_node = f"P::{st_id}::{p}"
            kg.add_node(pad_node, kind="pad", pad_type=st_data.get("pad_type"),
                        voltage=st_data.get("charging_voltage"))
            kg.add_edge(pad_node, node, relation="BELONGTO")

    for u, v, data in sky.graph.edges(data=True):
        kg.add_edge(f"S::{u}", f"S::{v}", relation="SkySegment", distance=data["distance"])
        kg.add_edge(f"S::{v}", f"S::{u}", relation="SkySegment", distance=data["distance"])

    for d in sky.drones:
        d_node = f"D::{d.id}"
        kg.add_node(d_node, kind="drone", payload=d.payload, battery=d.battery,
                    range_km=d.range_km, speed_kmh=d.speed_kmh)
        home = f"S::{d.home_station}"
        if home in kg:
            kg.add_edge(d_node, home, relation="ATTEND")

    for row in daas_catalog:
        daas_node = f"DaaS::{row['id']}"
        kg.add_node(daas_node, kind="daas", operation=row["operation"], price=row["price"])
        d_node = f"D::{row['drone_id']}"
        if d_node in kg:
            kg.add_edge(d_node, daas_node, relation="PROVIDE")

    # Similarity edges (D-SIMILAR, S-SIMILAR): pairwise feature closeness,
    # capped to keep the graph tractable (sampled candidate pairs).
    drone_nodes = [n for n, d in kg.nodes(data=True) if d.get("kind") == "drone"]
    for a, b in itertools.islice(itertools.combinations(drone_nodes, 2), 4000):
        if _feature_close(kg.nodes[a], kg.nodes[b], ["payload", "battery", "range_km", "speed_kmh"]):
            kg.add_edge(a, b, relation="D-SIMILAR")
            kg.add_edge(b, a, relation="D-SIMILAR")

    station_nodes = [n for n, d in kg.nodes(data=True) if d.get("kind") == "station"]
    for a, b in itertools.islice(itertools.combinations(station_nodes, 2), 6000):
        if _feature_close(kg.nodes[a], kg.nodes[b], ["pad_type", "charging_voltage"]):
            kg.add_edge(a, b, relation="R-SIMILAR")
            kg.add_edge(b, a, relation="R-SIMILAR")

    return kg
