"""Experiment 3 - Energy efficiency and failure rate (DaaS-GC.tex Section
7.3.3).

(a) Vary k in [2, 10], record percentage energy consumption / battery
depletion for DaaS-GC and DaaS-HGC.
(b) Fix k=5, vary payload over [1000, 5000] g, record average failure
rate for all four compared methods.
"""

from __future__ import annotations

import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.build_dataset import ITEMS_PER_REQUEST, _split_weight
from daasgc.runner import run_daas_gc, run_daas_hgc
from experiments.common import FIGURES_DIR, RESULTS_DIR, load_instance, run_all_methods

K_VALUES = list(range(2, 11))
WEIGHTS_G = [1000, 2000, 3000, 4000, 5000]
ENERGY_PER_KM_KWH = 0.0025   # nominal drone energy draw per km, used only for the energy-% metric


def _energy_percent(sky, path_stations, formation) -> float:
    if formation is None or len(path_stations) < 2:
        return 0.0
    distance = sum(sky.segment_distance(path_stations[i], path_stations[i + 1])
                    for i in range(len(path_stations) - 1))
    used_wh = distance * ENERGY_PER_KM_KWH * 1000
    avg_battery = sum(d.battery for d in formation.drones) / len(formation.drones)
    return round(min(100.0, 100 * used_wh / max(avg_battery, 1e-6)), 3)


def run() -> dict:
    sky, requests, catalog = load_instance(100, 20)
    req = requests[0]
    src, tgt, items, weights = req["source"], req["target"], req["items"], req["weights"]

    energy_gc, energy_hgc = [], []
    for k in K_VALUES:
        r_gc = run_daas_gc(sky, src, tgt, items, weights, target_k=k)
        energy_gc.append(_energy_percent(sky, r_gc.path.stations, r_gc.formation))
        r_hgc = run_daas_hgc(sky, src, tgt, items, weights, target_k=k)
        energy_hgc.append(_energy_percent(sky, r_hgc.path.stations, r_hgc.formation))

    # Failure-rate sweep across payloads, k fixed to 5.
    rng = random.Random(11)
    stations_ids = list(sky.graph.nodes)
    source, target = rng.sample(stations_ids, 2)
    methods = ["DaaS-GC", "DaaS-HGC", "DaaS-SG", "DaaS-KGE"]
    failure_by_weight = {m: [] for m in methods}
    for w_g in WEIGHTS_G:
        item_kg = _split_weight(w_g / 1000.0, ITEMS_PER_REQUEST, rng)
        request = {
            "source": source, "target": target,
            "items": [{"id": f"it{i+1}", "weight": round(w, 3)} for i, w in enumerate(item_kg)],
            "weights": {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25},
        }
        out = run_all_methods(sky, catalog, request, target_k=5)
        for m in methods:
            failure_by_weight[m].append(out[m]["failure_rate"])

    payload = {
        "k_values": K_VALUES, "energy_gc": energy_gc, "energy_hgc": energy_hgc,
        "weights_g": WEIGHTS_G, "failure_by_weight": failure_by_weight,
    }
    with open(os.path.join(RESULTS_DIR, "exp3_energy_failure.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _plot(payload)
    return payload


def _plot(payload: dict) -> None:
    k_values = payload["k_values"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    x = list(range(len(k_values)))
    ax.bar([xi - width / 2 for xi in x], payload["energy_gc"], width=width, label="DaaS-GC")
    ax.bar([xi + width / 2 for xi in x], payload["energy_hgc"], width=width, label="DaaS-HGC")
    ax.set_xticks(x, k_values)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Energy consumption (%)")
    ax.set_title("Energy consumption under varying numbers of clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_energy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, vals in payload["failure_by_weight"].items():
        ax.plot(payload["weights_g"], vals, marker="o", label=m)
    ax.set_xlabel("Package weight (g)")
    ax.set_ylabel("Average failure rate")
    ax.set_title("Impact of delivery query and clustering (k=5) on failure rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_failure_rate.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([payload["energy_gc"], payload["energy_hgc"]], labels=["DaaS-GC", "DaaS-HGC"])
    ax.set_ylabel("Energy consumption (%)")
    ax.set_title("Distribution of energy consumption across k in [2,10]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_energy_boxplot.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2)[:2000])
