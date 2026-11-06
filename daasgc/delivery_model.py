"""Delivery model shared by every method: drone flight/energy simulation,
delivery-plan evaluation (time, cost, energy) and Monte-Carlo failure rate.

Physical assumptions (documented in README, "Delivery model"):
* energy per km grows linearly with total mass (empty mass + load);
* planning keeps a 20 % battery reserve; at an intermediate station a drone
  recharges (to full) only if it cannot reach its next charging opportunity
  without touching the reserve, and only on a compatible pad type;
* charging time = 60 min x (restored energy / capacity) x voltage factor;
* drones of a formation fly as a convoy; at a station the drones that need
  to charge share the station's free pads (sequential charging when there
  are more drones than free pads), and the convoy leaves when all are done;
* delivery cost = sum_d [service price(d) + cost_per_km(d) x distance]
  + charging-session fees ("aggregate cost of the selected drone services
  and the recharging expenses", Section 6.3.1);
* failure rate = share of Monte-Carlo trials in which a drone fails
  (availability), runs out of energy (wind-dependent consumption), or the
  delivery time exceeds the SLA (1.2 x planned time) because of pad
  occupancy (binomial with the station's congestion) and waiting-time
  variability (uniform on [0, 2 x average waiting time]).
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from daasgc.sky_network import Drone, SkyNetwork

SLA_FACTOR = 1.2
RESERVE = 0.2              # planned flights keep 20 % of battery capacity in reserve
WIND_MAX_MS = 12.0          # range of wind speed in the CMU flight-log schema
WIND_SENSITIVITY = 0.02     # +2 % consumption per m/s of wind


@dataclass
class FlightTrace:
    feasible: bool
    charges: list[tuple[int, float, float]] = field(default_factory=list)  # (path index, Wh, minutes)
    energy_used: float = 0.0
    max_discharge: float = 0.0


def fly(sky: SkyNetwork, drone: Drone, load: float, path: list[str],
        consumption: list[float] | None = None) -> FlightTrace:
    """Simulate one drone along `path` with `load` kg.

    `consumption` optionally gives a per-segment multiplier (wind); the
    charging decisions are always taken on the nominal consumption.
    """
    if load > drone.payload + 1e-9:
        return FlightTrace(False)
    rate = drone.rate(load)
    seg_nominal = [sky.segment_distance(path[i], path[i + 1]) * rate for i in range(len(path) - 1)]
    seg_real = seg_nominal if consumption is None else [e * c for e, c in zip(seg_nominal, consumption)]
    n = len(path)
    charge_ok = [0 < i < n - 1 and sky.compatible(drone, path[i]) for i in range(n)]
    floor = RESERVE * drone.battery
    energy, used, since_full, max_dis = drone.battery, 0.0, 0.0, 0.0
    trace = FlightTrace(True)
    for i in range(n - 1):
        if charge_ok[i]:
            j = i + 1
            while j < n - 1 and not charge_ok[j]:
                j += 1
            need = sum(seg_nominal[i:j])
            if energy - need < floor - 1e-9:
                restored = drone.battery - energy
                trace.charges.append((i, restored, sky.charge_minutes(drone, path[i], restored)))
                energy, since_full = drone.battery, 0.0
        if seg_real[i] > energy - floor + 1e-9:
            return FlightTrace(False)
        energy -= seg_real[i]
        used += seg_real[i]
        since_full += seg_real[i]
        max_dis = max(max_dis, since_full)
    trace.energy_used, trace.max_discharge = used, max_dis
    return trace


def max_feasible_load(sky: SkyNetwork, drone: Drone, path: list[str], tol: float = 0.01) -> float:
    """Largest load (kg, <= payload) the drone can carry along `path` (bisection)."""
    if fly(sky, drone, drone.payload, path).feasible:
        return drone.payload
    if not fly(sky, drone, 0.0, path).feasible:
        return 0.0
    lo, hi = 0.0, drone.payload
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if fly(sky, drone, mid, path).feasible:
            lo = mid
        else:
            hi = mid
    return lo


def _station_makespan(durations: list[float], pads: int) -> float:
    """List scheduling of charging durations on `pads` identical pads."""
    if not durations:
        return 0.0
    heap = [0.0] * min(pads, len(durations))
    for d in sorted(durations, reverse=True):
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + d)
    return max(heap)


def plan_metrics(sky: SkyNetwork, path: list[str], drones: list[Drone], loads: dict[str, float],
                 traces: dict[str, FlightTrace] | None = None,
                 pads_override: dict[int, int] | None = None,
                 wait_override: dict[int, float] | None = None) -> dict:
    """Time (min), cost and energy figures of a delivery plan (path + formation)."""
    traces = traces or {d.id: fly(sky, d, loads[d.id], path) for d in drones}
    dist = sky.path_distance(path)
    speed = min(d.speed_kmh for d in drones)
    flight_min = 60.0 * dist / speed
    station_min, n_sessions, fees = 0.0, 0, 0.0
    by_station: dict[int, list[float]] = {}
    for d in drones:
        for idx, _wh, minutes in traces[d.id].charges:
            by_station.setdefault(idx, []).append(minutes)
    for idx, durations in by_station.items():
        sid = path[idx]
        pads = (pads_override or {}).get(idx, sky.free_pads(sid))
        wait = (wait_override or {}).get(idx, sky.waiting_time(sid))
        if pads <= 0:                      # every pad busy: wait for one more session
            wait += 0.5 * 60.0
            pads = 1
        station_min += wait + _station_makespan(durations, pads)
        n_sessions += len(durations)
        fees += len(durations) * sky.charging_cost(sid)
    service = sum(d.service_price for d in drones)
    cost = service + sum(d.cost_per_km * dist for d in drones) + fees
    ideal = sky.euclid(path[0], path[-1])
    used = sum(traces[d.id].energy_used for d in drones)
    ideal_energy = sum(d.rate(loads[d.id]) * ideal for d in drones)
    return {
        "delivery_time_min": flight_min + station_min,
        "delivery_cost": cost,
        "service_cost": service,
        "recharging_cost": fees,
        "distance": dist,
        "n_charging_sessions": n_sessions,
        "energy_used_wh": used,
        "energy_efficiency_pct": 100.0 * ideal_energy / used if used > 0 else 100.0,
        # battery depletion: deepest discharge between two full charges, % of capacity
        # ("depletion of nearly three-quarters of the battery capacity", Section 6.3.3)
        "battery_usage_pct": 100.0 * sum(traces[d.id].max_discharge / d.battery for d in drones) / len(drones),
        # total energy drawn over the trip, % of capacity (> 100 % when the drone recharged)
        "battery_energy_drawn_pct": 100.0 * sum(traces[d.id].energy_used / d.battery for d in drones) / len(drones),
    }


def failure_rate(sky: SkyNetwork, path: list[str], drones: list[Drone], loads: dict[str, float],
                 trials: int = 200, seed: int | str = 0) -> float:
    """Share (%) of Monte-Carlo trials in which the planned delivery fails."""
    rng = random.Random(seed)
    planned = plan_metrics(sky, path, drones, loads)["delivery_time_min"]
    nominal = {d.id: fly(sky, d, loads[d.id], path) for d in drones}
    failures = 0
    for _ in range(trials):
        if any(rng.random() > d.availability for d in drones):
            failures += 1
            continue
        wind = [1.0 + WIND_SENSITIVITY * rng.uniform(0.0, WIND_MAX_MS) for _ in range(len(path) - 1)]
        ok = True
        for d in drones:
            if not _replay(sky, d, loads[d.id], path, nominal[d.id], wind):
                ok = False
                break
        if not ok:
            failures += 1
            continue
        pads, waits = {}, {}
        for idx in {c[0] for t in nominal.values() for c in t.charges}:
            st = sky.station(path[idx])
            busy = sum(rng.random() < st["congestion"] for _ in range(st["n_pads"]))
            pads[idx] = st["n_pads"] - busy
            waits[idx] = rng.uniform(0.0, 2.0 * st["waiting_time"])     # mean = average waiting time
        real = plan_metrics(sky, path, drones, loads, traces=nominal,
                            pads_override=pads, wait_override=waits)["delivery_time_min"]
        if real > SLA_FACTOR * planned:
            failures += 1
    return 100.0 * failures / trials


def _replay(sky, drone, load, path, trace: FlightTrace, wind: list[float]) -> bool:
    """Fly the planned charging stops under real (wind) consumption."""
    stops = {c[0] for c in trace.charges}
    rate = drone.rate(load)
    energy = drone.battery
    for i in range(len(path) - 1):
        if i in stops:
            energy = drone.battery
        need = sky.segment_distance(path[i], path[i + 1]) * rate * wind[i]
        if need > energy + 1e-9:
            return False
        energy -= need
    return True
