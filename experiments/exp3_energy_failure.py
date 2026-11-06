"""Experiment 3 - Energy efficiency (Fig. 7, energy.png) and failure rate
(Fig. 8, failure-boxplots.png).

(a) 100 stations, 20 drones, 3000 g, k = 1..10: energy efficiency (ideal
straight-line energy / energy actually used) and battery usage (energy used /
battery capacity, > 100 % when drones recharge en route), DaaS-GC/DaaS-HGC,
with DaaS-SG and DaaS-KGE as k-independent references.
(b) k = 5, payload 1000..5000 g: failure rate of each plan, measured by
Monte-Carlo simulation (FAILURE_TRIALS trials per plan), four methods.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (DEFAULT_PAYLOAD_G, FAILURE_TRIALS, K_VALUES, METHODS, PAYLOADS_G,  # noqa: E402
                                load_instance, make_items, parse_args, pick_pair, query, record, rng_for,
                                save)

N_STATIONS, N_DRONES, K_FAIL = 100, 20, 5


def run(runs: int = 10, seed: int = 42, plot: bool = True) -> dict:
    inst = load_instance(N_STATIONS, N_DRONES)
    energy = {"DaaS-GC": {str(k): [] for k in K_VALUES}, "DaaS-HGC": {str(k): [] for k in K_VALUES},
              "DaaS-SG": [], "DaaS-KGE": []}
    for r in range(runs):
        rng = rng_for(seed, "exp3a", r)
        s, t = pick_pair(inst.sky, rng)
        items = make_items(DEFAULT_PAYLOAD_G, rng)
        for m in ("DaaS-SG", "DaaS-KGE"):
            energy[m].append(record(inst, query(inst, m, s, t, items, repeats=1)[0]))
        for k in K_VALUES:
            for m in ("DaaS-GC", "DaaS-HGC"):
                energy[m][str(k)].append(record(inst, query(inst, m, s, t, items, k=k, repeats=1)[0]))

    failure = {m: {str(w): [] for w in PAYLOADS_G} for m in METHODS}
    for r in range(runs):
        s, t = pick_pair(inst.sky, rng_for(seed, "exp3b", r))
        for w in PAYLOADS_G:
            items = make_items(w, rng_for(seed, "exp3b", r, w))
            for m in METHODS:
                out, _ = query(inst, m, s, t, items, k=K_FAIL, repeats=1)
                failure[m][str(w)].append(record(inst, out, failure_seed=f"{seed}|fail|{r}|{w}|{m}"))
    payload = {"experiment": "exp3_energy_failure", "figures": ["energy.png", "failure-boxplots.png"],
               "config": {"stations": N_STATIONS, "drones": N_DRONES, "k_energy": K_VALUES,
                          "k_failure": K_FAIL, "payload_g_energy": DEFAULT_PAYLOAD_G, "payloads_g_failure": PAYLOADS_G,
                          "failure_trials": FAILURE_TRIALS, "runs": runs, "seed": seed},
               "energy": energy, "failure": failure}
    save("exp3_energy_failure", payload)
    if plot:
        from plots.make_figures import fig_energy, fig_failure
        fig_energy(payload)
        fig_failure(payload)
    return payload


if __name__ == "__main__":
    a = parse_args(__doc__)
    run(a.runs, a.seed, not a.no_plot)
