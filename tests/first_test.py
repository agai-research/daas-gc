#!/usr/bin/env python3
"""Smoke test: builds a small instance and runs the four methods on a few requests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import METHODS, load_instance, make_items, pick_pair, query, record, rng_for  # noqa: E402


def main() -> None:
    inst = load_instance(50, 10)
    print(f"Instance s50_d10: {inst.sky.n_stations()} stations, {inst.sky.n_segments()} segments, "
          f"{len(inst.sky.drones)} drones")
    for r in range(3):
        rng = rng_for(0, "smoke", r)
        s, t = pick_pair(inst.sky, rng)
        items = make_items(3000, rng)
        print(f"\nRequest {r + 1}: {s} -> {t}, 3000 g")
        for m in METHODS:
            rec = record(inst, query(inst, m, s, t, items, k=5, repeats=1)[0])
            plan = "no feasible formation" if rec["failed"] else \
                f"{rec['n_drones']} drones, cost {rec['delivery_cost']:.1f}, time {rec['delivery_time_min']:.1f} min"
            print(f"  {m:9s} {len(rec['path'])} stations, {rec['distance']:6.1f} km, {plan}")
    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
