#!/usr/bin/env python3
"""Reproduce every result and figure of the experiments section.

    python main_exp.py --runs 10 --seed 42

1. builds (or reuses) the 18 dataset instances under data/instances/;
2. runs experiments 1-4 -> results/*.json;
3. draws the six figures with plots/make_figures.py -> figures/*.png.
"""

from __future__ import annotations

import argparse
import time

from data.build_instances import build_all
from experiments import exp1_user_request, exp2_partitioning, exp3_energy_failure, exp4_computation_time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", choices=["exp1", "exp2", "exp3", "exp4"])
    a = ap.parse_args()
    t0 = time.perf_counter()
    build_all()
    print(f"instances ready ({time.perf_counter() - t0:.1f}s)")
    for name, mod in (("exp1", exp1_user_request), ("exp2", exp2_partitioning),
                      ("exp3", exp3_energy_failure), ("exp4", exp4_computation_time)):
        if a.only and name not in a.only:
            continue
        t0 = time.perf_counter()
        mod.run(runs=a.runs, seed=a.seed, plot=True)
        print(f"{name} done ({time.perf_counter() - t0:.1f}s)")
    print("results/ and figures/ updated")


if __name__ == "__main__":
    main()
