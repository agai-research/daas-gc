"""Proximity-aware DaaS composition (DaaS_KGE-1.pdf, Algorithm 1, Eq. 11/12).

SCDaaS = (1/n) * sum(wk * Qk_i) over the four drone features considered in
the paper: total delivery cost, minimal flight range, minimal speed,
minimal battery capacity. Drones are sorted by this score, then greedily
packed with the requested items (same greedy-assignment shape as
Algorithm 1 of the PDF), producing candidate compositions that are scored
and ranked.
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.sky_network import Drone, SkyNetwork


@dataclass
class KGEFormation:
    drones: list[Drone]
    assignment: dict[str, list[str]]
    score: float


def scdaas_score(drones: list[Drone], weights: dict[str, float]) -> float:
    """Eq. 12 of DaaS_KGE-1.pdf:
    SCDaaS = (1/n) * (w1*sum(cost) + w2*min(speed) + w3*min(range) + w4*min(battery))
    Cost is a penalty (minimize), the other three are rewards (maximize),
    so the final ranking score subtracts the reward term.
    """
    if not drones:
        return float("inf")
    n = len(drones)
    w1, w2, w3, w4 = (weights.get(k, 0.25) for k in ("w1", "w2", "w3", "w4"))
    total_cost = sum(d.cost_per_km * 10 for d in drones)   # proxy per-mission cost
    min_speed = min(d.speed_kmh for d in drones)
    min_range = min(d.range_km for d in drones)
    min_battery = min(d.battery for d in drones)
    penalty = w1 * total_cost
    reward = w2 * min_speed + w3 * min_range + w4 * min_battery
    return (penalty - reward) / n


def compose_kge_formation(sky: SkyNetwork, candidate_drone_ids: list[str], items: list[dict],
                           weights: dict[str, float], max_segment: float) -> KGEFormation | None:
    id_to_drone = {d.id: d for d in sky.drones}
    pool = [id_to_drone[did] for did in candidate_drone_ids if did in id_to_drone]
    pool = [d for d in pool if d.range_km >= max_segment]
    if not pool:
        return None

    pool = sorted(pool, key=lambda d: scdaas_score([d], weights))

    best: KGEFormation | None = None
    remaining_pool = list(pool)
    while sum(d.payload for d in remaining_pool) >= sum(it["weight"] for it in items):
        pck = sorted(items, key=lambda it: it["weight"])
        formation: list[Drone] = []
        assignment: dict[str, list[str]] = {}
        working_pool = list(remaining_pool)
        while pck and working_pool:
            drone = working_pool.pop(0)
            capacity = drone.payload
            assigned, pending = [], []
            for it in pck:
                if it["weight"] <= capacity:
                    assigned.append(it["id"])
                    capacity -= it["weight"]
                else:
                    pending.append(it)
            pck = pending
            if assigned:
                formation.append(drone)
                assignment[drone.id] = assigned
        if not pck and formation:
            score = scdaas_score(formation, weights)
            candidate = KGEFormation(drones=formation, assignment=assignment, score=score)
            if best is None or candidate.score < best.score:
                best = candidate
            used = {d.id for d in formation}
            remaining_pool = [d for d in remaining_pool if d.id not in used]
            if not remaining_pool:
                break
        else:
            break
    return best
