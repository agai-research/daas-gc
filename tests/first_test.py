#!/usr/bin/env python3
"""tests/first_test.py - a dedicated smoke test of the DaaS-GC prototype.

Builds a small, self-contained sky network (kept separate from the
experiment instances), then runs all four methods on a handful of sample
delivery queries and prints a clear, human-readable summary. Use this to
check the prototype after cloning the repository, before running the
full experiment suite.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.build_dataset import build_and_save
from daasgc.runner import run_daas_gc, run_daas_hgc
from daasgc.sky_network import SkyNetwork
from baselines.daas_kge.run import run_daas_kge
from baselines.daas_sg import run_daas_sg

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_instance")


def build_sample() -> str:
    return build_and_save(n_stations=40, n_drones=12, out_dir=SAMPLE_DIR, seed=99)


def main() -> None:
    print("Building the sample knowledge/sky network for the first test...")
    instance_dir = build_sample()

    sky = SkyNetwork.from_json(
        os.path.join(instance_dir, "stations.json"),
        os.path.join(instance_dir, "segments.json"),
        os.path.join(instance_dir, "drones.json"),
    )
    with open(os.path.join(instance_dir, "daas_catalog.json")) as f:
        catalog = json.load(f)
    with open(os.path.join(instance_dir, "requests.json")) as f:
        requests = json.load(f)

    print(f"Sky network: {sky.n_stations()} stations, {sky.n_segments()} segments, "
          f"{len(sky.drones)} drones, {len(requests)} sample queries.\n")

    for req in requests[:3]:
        print(f"--- Query {req['id']}: {req['source']} -> {req['target']} "
              f"({req['package_weight_g']} g) ---")
        src, tgt, items, weights = req["source"], req["target"], req["items"], req["weights"]

        r_gc = run_daas_gc(sky, src, tgt, items, weights, target_k=5)
        r_hgc = run_daas_hgc(sky, src, tgt, items, weights, target_k=5)
        r_sg = run_daas_sg(sky, src, tgt, items, weights)
        r_kge = run_daas_kge(sky, catalog, src, tgt, items, weights, num_walks=3, walk_length=3, dimension=32)

        for name, r in (("DaaS-GC", r_gc), ("DaaS-HGC", r_hgc), ("DaaS-SG", r_sg), ("DaaS-KGE", r_kge)):
            path = r.path
            formation = r.formation
            n_drones = 0 if formation is None else len(formation.drones)
            print(f"  {name:10s} | path={'->'.join(path.stations):40s} | "
                  f"score={path.score:8.2f} | drones={n_drones}")
        print()

    print("First test complete: all four methods ran end to end without errors.")


if __name__ == "__main__":
    main()
