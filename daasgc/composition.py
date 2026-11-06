"""Drone formation selection (Algorithm 3, Eq. daas-score).

Runs after the delivery path is fixed (two-phase design). Drones are
filtered by what they can actually fly on this path (flight range under
load, pad-type compatibility of the stations where they must recharge),
sorted by payload, greedily packed with the requested items, and the
candidate formations are ranked with Eq. daas-score.

Sign convention of Eq. daas-score (made explicit): time and cost are
penalties, availability and reputation are rewards, so the formation that
MINIMISES  w1*T + w2*C - w3*prod(A) - w4*mean(R)  is selected.
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.delivery_model import max_feasible_load, plan_metrics
from daasgc.sky_network import Drone, SkyNetwork


@dataclass
class Formation:
    drones: list[Drone]
    assignment: dict[str, list[str]]        # drone id -> item ids
    loads: dict[str, float]                  # drone id -> kg
    score: float


def score_formation(sky: SkyNetwork, path: list[str], drones: list[Drone], loads: dict[str, float],
                    weights: dict[str, float]) -> float:
    m = plan_metrics(sky, path, drones, loads)
    avail = 1.0
    for d in drones:
        avail *= d.availability
    rep = sum(d.reputation for d in drones) / len(drones)
    w1, w2, w3, w4 = (weights.get(k, 0.25) for k in ("w1", "w2", "w3", "w4"))
    return w1 * m["delivery_time_min"] + w2 * m["delivery_cost"] - w3 * avail - w4 * rep


def pack(items: list[dict], pool: list[tuple[Drone, float]]):
    """Greedy packing of items (lightest first) into drones in pool order."""
    pck = sorted(items, key=lambda it: (it["weight"], it["id"]))
    formation, assignment, loads = [], {}, {}
    for drone, cap in pool:
        if not pck:
            break
        left, taken, pending = cap, [], []
        for it in pck:
            if it["weight"] <= left + 1e-9:
                taken.append(it["id"])
                left -= it["weight"]
            else:
                pending.append(it)
        pck = pending
        if taken:
            formation.append(drone)
            assignment[drone.id] = taken
            loads[drone.id] = cap - left
    return (formation, assignment, loads) if not pck else None


def candidate_pool(sky: SkyNetwork, path: list[str], items: list[dict],
                   drones: list[Drone] | None = None) -> list[tuple[Drone, float]]:
    lightest = min(it["weight"] for it in items)
    pool = []
    for d in (drones if drones is not None else sky.drones):
        cap = max_feasible_load(sky, d, path)
        if cap >= lightest:
            pool.append((d, cap))
    return pool


def enumerate_formations(sky, path, items, weights, pool, scorer):
    """Algorithm 3 loop: build a formation from the remaining drones, score
    it, remove its drones, repeat while the pool can still carry the items."""
    best = None
    total = sum(it["weight"] for it in items)
    remaining = list(pool)
    while remaining and sum(c for _, c in remaining) >= total - 1e-9:
        packed = pack(items, remaining)
        if packed is None:
            break
        formation, assignment, loads = packed
        score = scorer(formation, loads)
        if best is None or score < best.score:
            best = Formation(formation, assignment, loads, score)
        used = {d.id for d in formation}
        remaining = [(d, c) for d, c in remaining if d.id not in used]
    return best


def select_drone_formation(sky: SkyNetwork, path: list[str], items: list[dict],
                           weights: dict[str, float], drones: list[Drone] | None = None) -> Formation | None:
    pool = candidate_pool(sky, path, items, drones)
    pool.sort(key=lambda dc: (-dc[1], dc[0].id))          # sort by (feasible) payload
    return enumerate_formations(sky, path, items, weights, pool,
                                lambda f, loads: score_formation(sky, path, f, loads, weights))
