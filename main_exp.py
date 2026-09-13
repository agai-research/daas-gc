#!/usr/bin/env python3
"""main_exp.py - build every dataset instance, then run the four
experiment series (Section 7.3 of DaaS-GC.tex) end to end, saving JSON
results and figures.
"""

from __future__ import annotations

import time

from data.build_instances import build_all
from experiments import exp1_user_request, exp2_partitioning, exp3_energy_failure, exp4_computation_time


def main() -> None:
    print("Building dataset instances...")
    t0 = time.perf_counter()
    build_all()
    print(f"Dataset instances built in {time.perf_counter() - t0:.1f}s\n")

    for label, module in (
        ("Experiment 1 - impact of user request", exp1_user_request),
        ("Experiment 2 - impact of partitioning", exp2_partitioning),
        ("Experiment 3 - energy efficiency and failure rate", exp3_energy_failure),
        ("Experiment 4 - computational efficiency", exp4_computation_time),
    ):
        print(f"Running {label}...")
        t0 = time.perf_counter()
        module.run()
        print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    print("All experiments complete. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
