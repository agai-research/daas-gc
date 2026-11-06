"""Proximity-aware DaaS composition of DaaS-KGE (SCDaaS ranking, Eq. 11/12 of
the DaaS-KGE paper). Candidate drones come from the embedding subspace; they
are filtered with the same feasibility model as the other methods (flight
range under load, pad compatibility), sorted by SCDaaS and greedily packed.
"""

from __future__ import annotations

from daasgc.composition import Formation, candidate_pool, enumerate_formations
from daasgc.sky_network import Drone, SkyNetwork

KGEFormation = Formation


def scdaas_score(drones: list[Drone], weights: dict[str, float]) -> float:
    """(1/n)(w1*sum cost - w2*min speed - w3*min range - w4*min battery); lower is better."""
    if not drones:
        return float("inf")
    w1, w2, w3, w4 = (weights.get(k, 0.25) for k in ("w1", "w2", "w3", "w4"))
    penalty = w1 * sum(d.cost_per_km * 10 for d in drones)
    reward = w2 * min(d.speed_kmh for d in drones) + w3 * min(d.range_km for d in drones) \
        + w4 * min(d.battery for d in drones)
    return (penalty - reward) / len(drones)


def compose_kge_formation(sky: SkyNetwork, path: list[str], candidate_drone_ids: list[str],
                          items: list[dict], weights: dict[str, float]) -> Formation | None:
    by_id = {d.id: d for d in sky.drones}
    pool = candidate_pool(sky, path, items, [by_id[i] for i in candidate_drone_ids if i in by_id])
    pool.sort(key=lambda dc: (scdaas_score([dc[0]], weights), dc[0].id))
    return enumerate_formations(sky, path, items, weights, pool,
                                lambda f, loads: scdaas_score(f, weights))
