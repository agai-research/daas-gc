"""Experiment 1 - Impact of user request (DaaS-GC.tex Section 7.3.1).

Package weight varies over [1000, 2000, 3000, 4000, 5000] g with the
number of clusters fixed to k=5 for DaaS-GC/DaaS-HGC. Records formation
size, delivery cost and delivery time for all four compared methods.
"""

from __future__ import annotations

import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.build_dataset import ITEMS_PER_REQUEST, _split_weight
from experiments.common import FIGURES_DIR, RESULTS_DIR, load_instance, run_all_methods

METHODS = ["DaaS-GC", "DaaS-HGC", "DaaS-SG", "DaaS-KGE"]
WEIGHTS_G = [1000, 2000, 3000, 4000, 5000]


def build_weight_requests(stations, seed: int = 7) -> tuple[str, str, list[dict]]:
    rng = random.Random(seed)
    station_ids = [s["id"] for s in stations]
    source, target = rng.sample(station_ids, 2)
    per_weight_items = {}
    for w_g in WEIGHTS_G:
        item_kg = _split_weight(w_g / 1000.0, ITEMS_PER_REQUEST, rng)
        per_weight_items[w_g] = [{"id": f"it{i+1}", "weight": round(w, 3)} for i, w in enumerate(item_kg)]
    return source, target, per_weight_items


def run() -> dict:
    sky, requests, catalog = load_instance(100, 20)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "instances", "s100_d20", "stations.json")) as f:
        stations = json.load(f)

    source, target, per_weight_items = build_weight_requests(stations)
    weights = {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}

    results = {m: {"formation_size": [], "delivery_cost": [], "delivery_time_min": []} for m in METHODS}

    for w_g in WEIGHTS_G:
        request = {"source": source, "target": target, "items": per_weight_items[w_g], "weights": weights}
        method_out = run_all_methods(sky, catalog, request, target_k=5)
        for m in METHODS:
            rec = method_out[m]
            results[m]["formation_size"].append(rec["n_drones"])
            results[m]["delivery_cost"].append(rec["delivery_cost"])
            results[m]["delivery_time_min"].append(rec["delivery_time_min"])

    payload = {"weights_g": WEIGHTS_G, "results": results}
    with open(os.path.join(RESULTS_DIR, "exp1_user_request.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _plot(payload)
    return payload


def _plot(payload: dict) -> None:
    weights_g = payload["weights_g"]
    results = payload["results"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in METHODS:
        ax.plot(weights_g, results[m]["formation_size"], marker="o", label=m)
    ax.set_xlabel("Package weight (g)")
    ax.set_ylabel("Number of drones in formation")
    ax.set_title("Impact of package weight on formation size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp1_formation_size.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in METHODS:
        ax.plot(weights_g, results[m]["delivery_cost"], marker="s", label=m)
    ax.set_xlabel("Package weight (g)")
    ax.set_ylabel("Delivery cost")
    ax.set_title("Impact of package weight on delivery cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp1_delivery_cost.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 150
    x = weights_g
    for i, m in enumerate(METHODS):
        offs = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offs, results[m]["delivery_time_min"], width=width, label=m)
    ax.set_xlabel("Package weight (g)")
    ax.set_ylabel("Delivery time (min)")
    ax.set_title("Impact of package weight on delivery time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp1_delivery_time.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2)[:2000])
