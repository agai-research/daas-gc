#!/usr/bin/env python3
"""Run one method on one delivery request and print the delivery plan.

    python daasgc.py --stations 100 --drones 20 --method DaaS-GC --k 5 --payload 3000 --seed 1
"""

from __future__ import annotations

import argparse
import json

from experiments.common import METHODS, load_instance, make_items, pick_pair, query, record, rng_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", type=int, default=100)
    ap.add_argument("--drones", type=int, default=20)
    ap.add_argument("--method", default="DaaS-GC", choices=METHODS)
    ap.add_argument("--k", type=int, default=5, help="target number of clusters (DaaS-GC/HGC); 1 = single graph")
    ap.add_argument("--payload", type=float, default=3000, help="package weight (g)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the random request")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--output", help="optional JSON output file")
    a = ap.parse_args()

    inst = load_instance(a.stations, a.drones)
    rng = rng_for(a.seed, "cli")
    s, t = (a.source, a.target) if a.source and a.target else pick_pair(inst.sky, rng)
    items = make_items(a.payload, rng)
    res, ms = query(inst, a.method, s, t, items, k=a.k)
    rec = record(inst, res, ms)
    print(f"\n=== {a.method} on s{a.stations}_d{a.drones}: {s} -> {t}, {a.payload:.0f} g ===")
    if a.method in ("DaaS-GC", "DaaS-HGC"):
        from daasgc.runner import prepare
        print("Partition        :", prepare(inst.sky, a.k).clusters.summary())
    print("Scenario         :", rec["scenario"])
    print("Path             :", " -> ".join(rec["path"]))
    print(f"Distance (km)    : {rec['distance']:.2f}   path score: {rec['path_score']:.1f}")
    if rec["failed"]:
        print("Drone formation  : none feasible")
    else:
        print("Drone formation  :", rec["drones"], " loads (kg):",
              {d: round(res.formation.loads[d], 3) for d in rec["drones"]})
        print(f"Delivery cost    : {rec['delivery_cost']:.2f}   delivery time: {rec['delivery_time_min']:.1f} min")
    print(f"Query time       : {ms:.2f} ms")
    if a.output:
        with open(a.output, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
