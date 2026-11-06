"""Experiment 2 - Impact of partitioning (Fig. 6, stations-distance.png).

k = 1..10 for 50, 100 and 200 stations (20 drones, 3000 g). DaaS-GC and
DaaS-HGC are run for every k; DaaS-SG and DaaS-KGE (which do not depend on k)
are run once per request and drawn as reference lines. k = 1 is the
degenerate single-cluster partition, on which DaaS-GC and DaaS-HGC must
return exactly the DaaS-SG plan: this is checked for every run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daasgc.runner import prepare  # noqa: E402
from experiments.common import (DEFAULT_PAYLOAD_G, K_VALUES, load_instance, make_items, parse_args,  # noqa: E402
                                pick_pair, query, record, rng_for, same_plan, save)

STATION_SIZES = [50, 100, 200]
N_DRONES = 20


def run(runs: int = 10, seed: int = 42, plot: bool = True) -> dict:
    out = {}
    for n in STATION_SIZES:
        inst = load_instance(n, N_DRONES)
        part_info = {str(k): prepare(inst.sky, k).clusters.summary() for k in K_VALUES}
        res = {"DaaS-GC": {str(k): [] for k in K_VALUES}, "DaaS-HGC": {str(k): [] for k in K_VALUES},
               "DaaS-SG": [], "DaaS-KGE": []}
        for r in range(runs):
            rng = rng_for(seed, "exp2", n, r)
            s, t = pick_pair(inst.sky, rng)
            items = make_items(DEFAULT_PAYLOAD_G, rng)
            sg = record(inst, query(inst, "DaaS-SG", s, t, items, repeats=1)[0])
            res["DaaS-SG"].append(sg)
            res["DaaS-KGE"].append(record(inst, query(inst, "DaaS-KGE", s, t, items, repeats=1)[0]))
            for k in K_VALUES:
                for m in ("DaaS-GC", "DaaS-HGC"):
                    rec = record(inst, query(inst, m, s, t, items, k=k, repeats=1)[0])
                    if k == 1 and not same_plan(rec, sg):
                        raise AssertionError(f"{m} with k=1 differs from DaaS-SG (n={n}, run={r})")
                    res[m][str(k)].append(rec)
        out[str(n)] = {"partitions": part_info, "results": res}
    payload = {"experiment": "exp2_partitioning", "figure": "stations-distance.png",
               "config": {"stations": STATION_SIZES, "drones": N_DRONES, "k": K_VALUES,
                          "payload_g": DEFAULT_PAYLOAD_G, "runs": runs, "seed": seed},
               "k1_equals_sg_checked": True, "sizes": out}
    save("exp2_partitioning", payload)
    if plot:
        from plots.make_figures import fig_stations_distance
        fig_stations_distance(payload)
    return payload


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.runs, a.seed, not a.no_plot)
