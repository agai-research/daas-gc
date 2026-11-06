"""Experiment 1 - Impact of user request (Fig. 5, drone-cost-time.png).

100 stations, 20 drones, k = 5 for DaaS-GC/DaaS-HGC. Package weight 1000..5000 g
(8 items). Per run: one source/target pair, the five payloads, four methods.
Records formation size, delivery cost and delivery time.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (METHODS, PAYLOADS_G, load_instance, make_items, parse_args,  # noqa: E402
                                pick_pair, query, record, rng_for, save)

N_STATIONS, N_DRONES, K = 100, 20, 5


def run(runs: int = 10, seed: int = 42, plot: bool = True) -> dict:
    inst = load_instance(N_STATIONS, N_DRONES)
    res = {m: {str(w): [] for w in PAYLOADS_G} for m in METHODS}
    for r in range(runs):
        s, t = pick_pair(inst.sky, rng_for(seed, "exp1", r))
        for w in PAYLOADS_G:
            items = make_items(w, rng_for(seed, "exp1", r, w))
            for m in METHODS:
                out, ms = query(inst, m, s, t, items, k=K, repeats=1)
                rec = record(inst, out, ms)
                rec.update({"run": r, "source": s, "target": t})
                res[m][str(w)].append(rec)
    payload = {"experiment": "exp1_user_request", "figure": "drone-cost-time.png",
               "config": {"stations": N_STATIONS, "drones": N_DRONES, "k": K, "payloads_g": PAYLOADS_G,
                          "runs": runs, "seed": seed, "qos_weights": "balanced 0.25"},
               "results": res}
    save("exp1_user_request", payload)
    if plot:
        from plots.make_figures import fig_drone_cost_time
        fig_drone_cost_time(payload)
    return payload


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.runs, a.seed, not a.no_plot)
