"""Experiment 4 - Computational efficiency (Fig. 9 comp-time-clusters-boxplots.png,
Fig. 10 comp-time-sn-boxplots.png).

Timed = query only (path search + formation selection), median of 3
executions, ms. Pre-processing (calibration/clustering/hyper-graph for
DaaS-GC/HGC, KG + embedding for DaaS-KGE) is measured separately and reported
in the JSON, never inside the query time.

(a) 50/100/200 stations, 20 drones, 3000 g, k = 1..10: DaaS-GC, DaaS-HGC;
    DaaS-SG and DaaS-KGE (k-independent) as references.
(b) k = 5, stations 50..300 x drones 10/20/30, 3000 g: all four methods.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.daas_kge.run import prepare_kge  # noqa: E402
from daasgc.clustering import calibration_sweep  # noqa: E402
from daasgc.runner import prepare  # noqa: E402
from experiments.common import (DEFAULT_PAYLOAD_G, K_VALUES, METHODS, Instance, load_instance,  # noqa: E402
                                make_items, parse_args, pick_pair, query, rng_for, save)

SIZES_A = [50, 100, 200]
SIZES_B = [50, 100, 150, 200, 250, 300]
FLEETS = [10, 20, 30]
K_B = 5


def preprocessing_ms(n: int, d: int, ks=(1, 5, 10)) -> dict:
    """Pre-processing cost measured on a fresh copy of the instance."""
    inst = Instance(n, d)
    sky = inst.sky
    t0 = time.perf_counter()
    calibration_sweep(sky)
    sweep_ms = (time.perf_counter() - t0) * 1000
    part_ms = {}
    for k in ks:
        t0 = time.perf_counter()
        prepare(sky, k)
        part_ms[str(k)] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    prepare_kge(sky, inst.catalog)
    return {"calibration_sweep_ms": sweep_ms, "partition_and_hypergraph_ms": part_ms,
            "kge_kg_and_embedding_ms": (time.perf_counter() - t0) * 1000}


def run(runs: int = 10, seed: int = 42, plot: bool = True) -> dict:
    part_a = {}
    for n in SIZES_A:
        inst = load_instance(n, 20)
        res = {"DaaS-GC": {str(k): [] for k in K_VALUES}, "DaaS-HGC": {str(k): [] for k in K_VALUES},
               "DaaS-SG": [], "DaaS-KGE": []}
        for k in K_VALUES:
            prepare(inst.sky, k)
        inst.kge()
        for r in range(runs):
            rng = rng_for(seed, "exp4a", n, r)
            s, t = pick_pair(inst.sky, rng)
            items = make_items(DEFAULT_PAYLOAD_G, rng)
            for m in ("DaaS-SG", "DaaS-KGE"):
                res[m].append(query(inst, m, s, t, items)[1])
            for k in K_VALUES:
                for m in ("DaaS-GC", "DaaS-HGC"):
                    res[m][str(k)].append(query(inst, m, s, t, items, k=k)[1])
        part_a[str(n)] = res

    part_b, prep = {}, {}
    for d in FLEETS:
        part_b[str(d)] = {m: {} for m in METHODS}
        for n in SIZES_B:
            inst = load_instance(n, d)
            prepare(inst.sky, K_B)
            inst.kge()
            for m in METHODS:
                part_b[str(d)][m][str(n)] = []
            for r in range(runs):
                rng = rng_for(seed, "exp4b", n, d, r)
                s, t = pick_pair(inst.sky, rng)
                items = make_items(DEFAULT_PAYLOAD_G, rng)
                for m in METHODS:
                    part_b[str(d)][m][str(n)].append(query(inst, m, s, t, items, k=K_B)[1])
            if d == 20:
                prep[str(n)] = preprocessing_ms(n, d)

    payload = {"experiment": "exp4_computation_time",
               "figures": ["comp-time-clusters-boxplots.png", "comp-time-sn-boxplots.png"],
               "config": {"timed": "query = path search + formation selection (median of 3), ms",
                          "sizes_a": SIZES_A, "k_a": K_VALUES, "sizes_b": SIZES_B, "fleets_b": FLEETS,
                          "k_b": K_B, "payload_g": DEFAULT_PAYLOAD_G, "runs": runs, "seed": seed},
               "time_vs_k_ms": part_a, "time_vs_stations_ms": part_b, "preprocessing_ms": prep}
    save("exp4_computation_time", payload)
    if plot:
        from plots.make_figures import fig_comp_time_clusters, fig_comp_time_sn
        fig_comp_time_clusters(payload)
        fig_comp_time_sn(payload)
    return payload


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.runs, a.seed, not a.no_plot)
