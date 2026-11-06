"""Shared helpers for the experiment scripts.

Protocol (applies to every experiment):
* a *run* is one random delivery request (source/target pair not directly
  linked by a segment, 8 items splitting the package weight), drawn from a
  generator seeded with (--seed, experiment, run, ...); --runs runs per
  configuration (default 10);
* pre-processing is done once per network and is NOT timed for any method:
  (eps, mu) calibration + clustering + regions + hyper-graph for
  DaaS-GC/DaaS-HGC, knowledge-graph build + embedding training for DaaS-KGE;
* the timed part (computation time) is the query: delivery path search +
  drone formation selection, median of TIMING_REPEATS executions, in ms.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.daas_kge.run import prepare_kge, query_daas_kge  # noqa: E402
from baselines.daas_sg import run_daas_sg  # noqa: E402
from data.build_dataset import ITEMS_PER_REQUEST, ensure_instance  # noqa: E402
from daasgc.delivery_model import failure_rate, plan_metrics  # noqa: E402
from daasgc.runner import prepare, query_daas_gc, query_daas_hgc  # noqa: E402
from daasgc.sky_network import SkyNetwork  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

METHODS = ["DaaS-GC", "DaaS-HGC", "DaaS-KGE", "DaaS-SG"]
BALANCED = {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}
PAYLOADS_G = [1000, 2000, 3000, 4000, 5000]
DEFAULT_PAYLOAD_G = 3000
K_VALUES = list(range(1, 11))
TIMING_REPEATS = 3
FAILURE_TRIALS = 200


class Instance:
    def __init__(self, n_stations: int, n_drones: int):
        d = ensure_instance(n_stations, n_drones)
        self.n_stations, self.n_drones = n_stations, n_drones
        self.sky = SkyNetwork.from_dir(d)
        with open(os.path.join(d, "daas_catalog.json")) as f:
            self.catalog = json.load(f)

    def kge(self):
        return prepare_kge(self.sky, self.catalog)


@lru_cache(maxsize=None)
def load_instance(n_stations: int, n_drones: int) -> Instance:
    return Instance(n_stations, n_drones)


def rng_for(seed: int, *tags) -> random.Random:
    return random.Random("|".join(str(t) for t in (seed,) + tags))


def pick_pair(sky: SkyNetwork, rng: random.Random) -> tuple[str, str]:
    ids = list(sky.graph.nodes)
    while True:
        s, t = rng.sample(ids, 2)
        if not sky.graph.has_edge(s, t):
            return s, t


def make_items(payload_g: float, rng: random.Random) -> list[dict]:
    raw = [rng.random() + 0.2 for _ in range(ITEMS_PER_REQUEST)]
    total = payload_g / 1000.0
    return [{"id": f"it{i + 1}", "weight": round(total * r / sum(raw), 4)} for i, r in enumerate(raw)]


def query(inst: Instance, method: str, s: str, t: str, items, weights=BALANCED, k: int = 5,
          repeats: int = TIMING_REPEATS):
    """Run one method on one request; returns (result, query time in ms)."""
    sky = inst.sky
    if method in ("DaaS-GC", "DaaS-HGC"):
        part = prepare(sky, k)                       # pre-processing (cached, not timed)
        fn = (query_daas_gc if method == "DaaS-GC" else query_daas_hgc)
        call = lambda: fn(sky, part, s, t, items, weights)  # noqa: E731
    elif method == "DaaS-KGE":
        kge = inst.kge()                             # pre-processing (cached, not timed)
        call = lambda: query_daas_kge(sky, kge, s, t, items, weights)  # noqa: E731
    elif method == "DaaS-SG":
        call = lambda: run_daas_sg(sky, s, t, items, weights)  # noqa: E731
    else:
        raise ValueError(method)
    times, res = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        res = call()
        times.append((time.perf_counter() - t0) * 1000.0)
    return res, statistics.median(times)


def record(inst: Instance, res, ms: float | None = None, failure_seed=None) -> dict:
    sky, path, f = inst.sky, res.path, res.formation
    rec = {"path": path.stations, "n_stations": len(path.stations),
           "distance": round(sky.path_distance(path.stations), 4),
           "path_score": round(path.score, 4), "scenario": path.scenario,
           "fallback": path.fallback, "astar_calls": path.astar_calls,
           "time_ms": None if ms is None else round(ms, 4), "failed": f is None}
    if f is not None:
        m = plan_metrics(sky, path.stations, f.drones, f.loads)
        rec.update({"n_drones": len(f.drones), "drones": [d.id for d in f.drones],
                    **{k: round(v, 4) for k, v in m.items()}})
        if failure_seed is not None:
            rec["failure_rate_pct"] = failure_rate(sky, path.stations, f.drones, f.loads,
                                                   trials=FAILURE_TRIALS, seed=failure_seed)
    elif failure_seed is not None:
        rec["failure_rate_pct"] = 100.0
    return rec


def save(name: str, payload: dict) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--runs", type=int, default=10, help="requests per configuration (default 10)")
    ap.add_argument("--seed", type=int, default=42, help="base seed of the request generator")
    ap.add_argument("--no-plot", action="store_true", help="only write results/*.json")
    return ap.parse_args()


def same_plan(a: dict, b: dict) -> bool:
    keys = ("path", "drones", "delivery_cost", "delivery_time_min")
    return all(a.get(k) == b.get(k) for k in keys)
