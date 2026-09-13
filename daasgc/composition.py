"""Drone formation selection (DaaS-GC.tex, Algorithm 3, Eq. "eq:daas-score").

Runs strictly *after* the delivery path has been fixed by path_search.py
(two-phase design): filters drones by flight range, greedily packs items
into drones sorted by payload, then ranks the resulting candidate
formations with the QoS scoring function that trades off time, cost,
availability and reputation according to user-supplied weights w1..w4.
"""

from __future__ import annotations

from dataclasses import dataclass

from daasgc.sky_network import Drone, SkyNetwork


@dataclass
class Formation:
    drones: list[Drone]
    assignment: dict[str, list[str]]   # drone id -> item ids
    score: float


def _time_at_station(sky: SkyNetwork, drone: Drone, station: str) -> float:
    """t_ij: waiting + charging time for drone at a station."""
    return sky.waiting_time(station) + sky.charging_cost(station) / max(drone.speed_kmh, 1e-6)


def _cost_at_station(sky: SkyNetwork, drone: Drone, station: str, dist: float) -> float:
    """cost_ij * dist_ij contribution."""
    return drone.cost_per_km * dist


def score_formation(sky: SkyNetwork, path_stations: list[str], formation: list[Drone],
                     weights: dict[str, float]) -> float:
    """Eq. "eq:daas-score":
    score(F) = w1*max_time + w2*sum(cost*dist) + w3*prod(availability) + w4*mean(reputation)
    Lower time/cost is better, higher availability/reputation is better, so
    we minimize (w1*time + w2*cost) and subtract the (w3*avail + w4*reput)
    reward term to obtain a single minimization objective.
    """
    if not formation:
        return float("inf")
    r = max(len(path_stations) - 1, 1)
    total_dist = sum(sky.segment_distance(path_stations[i], path_stations[i + 1])
                      for i in range(len(path_stations) - 1))

    per_drone_time = []
    total_cost = 0.0
    for d in formation:
        t = sum(_time_at_station(sky, d, rs) for rs in path_stations)
        t += total_dist / max(d.speed_kmh, 1e-6)
        per_drone_time.append(t)
        total_cost += _cost_at_station(sky, d, path_stations[-1], total_dist)

    max_time = max(per_drone_time)
    availability = 1.0
    for d in formation:
        availability *= d.availability
    reputation = sum(d.reputation for d in formation) / len(formation)

    w1, w2, w3, w4 = (weights.get(k, 0.25) for k in ("w1", "w2", "w3", "w4"))
    penalty = w1 * max_time + w2 * total_cost
    reward = w3 * availability + w4 * reputation
    return penalty - reward


def select_drone_formation(sky: SkyNetwork, path_stations: list[str], items: list[dict],
                            weights: dict[str, float]) -> Formation | None:
    """Algorithm 3: filter by range, sort by payload, greedily pack items,
    generate candidate formations, score with Eq. "eq:daas-score", return
    the best (lowest-score) formation.
    """
    max_segment = max((sky.segment_distance(path_stations[i], path_stations[i + 1])
                        for i in range(len(path_stations) - 1)), default=0.0)

    candidates = [d for d in sky.drones if d.range_km >= max_segment]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda d: d.payload, reverse=True)

    best_formation: Formation | None = None
    remaining_pool = list(candidates)

    while sum(d.payload for d in remaining_pool) >= sum(it["weight"] for it in items):
        pool = list(remaining_pool)
        pck = sorted(items, key=lambda it: it["weight"])
        formation: list[Drone] = []
        assignment: dict[str, list[str]] = {}
        while pck and pool:
            drone = pool.pop(0)
            remaining_capacity = drone.payload
            assigned_items = []
            still_pending = []
            for it in pck:
                if it["weight"] <= remaining_capacity:
                    assigned_items.append(it["id"])
                    remaining_capacity -= it["weight"]
                else:
                    still_pending.append(it)
            pck = still_pending
            if assigned_items:
                formation.append(drone)
                assignment[drone.id] = assigned_items

        if not pck and formation:
            score = score_formation(sky, path_stations, formation, weights)
            candidate = Formation(drones=formation, assignment=assignment, score=score)
            if best_formation is None or candidate.score < best_formation.score:
                best_formation = candidate
            # remove the used drones from the pool to explore the next
            # candidate formation with the remaining drones (loop guard).
            used_ids = {d.id for d in formation}
            remaining_pool = [d for d in remaining_pool if d.id not in used_ids]
            if not remaining_pool:
                break
        else:
            break

    return best_formation
