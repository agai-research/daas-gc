#!/usr/bin/env python3
"""DaaS-GC.py - main entry point of the prototype.

Runs one method (DaaS-GC, DaaS-HGC, DaaS-SG, or DaaS-KGE) on a chosen sky
network instance and a chosen delivery request, then prints a clear,
human-readable result (selected path, drone formation, scores) and
optionally saves it to a JSON output file.

Usage:
    python daasgc.py --instance data/instances/s100_d20 --request req1 --method DaaS-GC
"""

from __future__ import annotations

import argparse
import json
import os

from daasgc.runner import run_daas_gc, run_daas_hgc
from daasgc.sky_network import SkyNetwork
from baselines.daas_kge.run import run_daas_kge
from baselines.daas_sg import run_daas_sg


def load_request(instance_dir: str, request_id: str | None) -> dict:
    with open(os.path.join(instance_dir, "requests.json")) as f:
        requests = json.load(f)
    if request_id is None:
        return requests[0]
    for r in requests:
        if r["id"] == request_id:
            return r
    raise ValueError(f"Request {request_id} not found in {instance_dir}/requests.json")


def run_method(method: str, sky: SkyNetwork, catalog: list[dict], request: dict, target_k: int):
    src, tgt, items, weights = request["source"], request["target"], request["items"], request["weights"]
    if method == "DaaS-GC":
        r = run_daas_gc(sky, src, tgt, items, weights, target_k=target_k)
        return r.path, r.formation, {"k": r.cluster_result.k, "eps": r.eps, "mu": r.mu}
    if method == "DaaS-HGC":
        r = run_daas_hgc(sky, src, tgt, items, weights, target_k=target_k)
        return r.path, r.formation, {"k": r.cluster_result.k, "eps": r.eps, "mu": r.mu}
    if method == "DaaS-SG":
        r = run_daas_sg(sky, src, tgt, items, weights)
        return r.path, r.formation, {}
    if method == "DaaS-KGE":
        r = run_daas_kge(sky, catalog, src, tgt, items, weights)
        return r.path, r.formation, {}
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DaaS-GC prototype on one delivery request.")
    parser.add_argument("--instance", default="data/instances/s100_d20",
                        help="Path to a dataset instance folder (stations/segments/drones/requests JSON).")
    parser.add_argument("--request", default=None, help="Request id to run (defaults to the first one).")
    parser.add_argument("--method", default="DaaS-GC",
                        choices=["DaaS-GC", "DaaS-HGC", "DaaS-SG", "DaaS-KGE"])
    parser.add_argument("--k", type=int, default=5, help="Target number of clusters (DaaS-GC/DaaS-HGC only).")
    parser.add_argument("--out", default=None, help="Optional path to save the result as JSON.")
    args = parser.parse_args()

    sky = SkyNetwork.from_json(
        os.path.join(args.instance, "stations.json"),
        os.path.join(args.instance, "segments.json"),
        os.path.join(args.instance, "drones.json"),
    )
    with open(os.path.join(args.instance, "daas_catalog.json")) as f:
        catalog = json.load(f)
    request = load_request(args.instance, args.request)

    path, formation, extra = run_method(args.method, sky, catalog, request, args.k)

    print(f"\n=== {args.method} result for request {request['id']} "
          f"({request['source']} -> {request['target']}) ===")
    print(f"Scenario         : {getattr(path, 'scenario', 'n/a')}")
    print(f"Selected path    : {' -> '.join(path.stations)}")
    print(f"Path score       : {path.score:.3f}")
    if extra:
        print(f"Clustering       : k={extra.get('k')}  eps={extra.get('eps')}  mu={extra.get('mu')}")
    if formation is None:
        print("Drone formation  : NONE (no drone combination could fulfil this request)")
    else:
        drone_ids = [d.id for d in formation.drones]
        print(f"Drone formation  : {drone_ids}")
        print(f"Formation score  : {formation.score:.3f}")
        print(f"Item assignment  : {formation.assignment}")

    result = {
        "method": args.method,
        "request_id": request["id"],
        "path": path.stations,
        "path_score": path.score,
        "scenario": getattr(path, "scenario", None),
        "formation": None if formation is None else {
            "drones": [d.id for d in formation.drones],
            "score": formation.score,
            "assignment": formation.assignment,
        },
        **extra,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result to {args.out}")


if __name__ == "__main__":
    main()
