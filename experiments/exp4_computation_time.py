"""Experiment 4 - Computational efficiency (DaaS-GC.tex Section 7.3.4).

(a) Fixed sky-network size (100 stations), vary k in [2, 10]: record
computation time for DaaS-GC and DaaS-HGC.
(b) k fixed to 5 for DaaS-GC/DaaS-HGC, vary the number of stations over
[50, 300] across three drone-fleet sizes (10, 20, 30): record computation
time for all four compared methods.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daasgc.runner import run_daas_gc, run_daas_hgc
from experiments.common import FIGURES_DIR, RESULTS_DIR, load_instance, run_all_methods

K_VALUES = list(range(2, 11))
STATION_SIZES = [50, 100, 150, 200, 250, 300]
FLEET_SIZES = [10, 20, 30]


def _time_only(fn, *args, **kwargs) -> float:
    import time
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def run() -> dict:
    # (a) drones sweep: fixed 100-station instance, k in [2,10].
    sky, requests, catalog = load_instance(100, 20)
    req = requests[0]
    src, tgt, items, weights = req["source"], req["target"], req["items"], req["weights"]

    time_gc_k, time_hgc_k = [], []
    for k in K_VALUES:
        time_gc_k.append(_time_only(run_daas_gc, sky, src, tgt, items, weights, target_k=k))
        time_hgc_k.append(_time_only(run_daas_hgc, sky, src, tgt, items, weights, target_k=k))

    # (b) stations sweep for three fleet sizes, all four methods, k=5.
    station_results = {str(fleet): {"DaaS-GC": [], "DaaS-HGC": [], "DaaS-SG": [], "DaaS-KGE": []}
                        for fleet in FLEET_SIZES}
    for fleet in FLEET_SIZES:
        for n_stations in STATION_SIZES:
            sky_s, requests_s, catalog_s = load_instance(n_stations, fleet)
            req_s = requests_s[0]
            out = run_all_methods(sky_s, catalog_s, req_s, target_k=5,
                                   kge_walks=2, kge_walk_len=3, kge_dim=24)
            for m in ("DaaS-GC", "DaaS-HGC", "DaaS-SG", "DaaS-KGE"):
                station_results[str(fleet)][m].append(round(out[m]["computation_time_s"], 5))

    payload = {
        "k_values": K_VALUES, "time_gc_k": time_gc_k, "time_hgc_k": time_hgc_k,
        "station_sizes": STATION_SIZES, "fleet_sizes": FLEET_SIZES,
        "station_results": station_results,
    }
    with open(os.path.join(RESULTS_DIR, "exp4_computation_time.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _plot(payload)
    return payload


def _plot(payload: dict) -> None:
    k_values = payload["k_values"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k_values, payload["time_gc_k"], marker="o", label="DaaS-GC")
    ax.plot(k_values, payload["time_hgc_k"], marker="s", label="DaaS-HGC")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Computation time (s)")
    ax.set_title("Computation time vs. number of clusters (100 stations)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp4_time_vs_k.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(FLEET_SIZES), figsize=(5 * len(FLEET_SIZES), 4.5), sharey=True)
    for ax, fleet in zip(axes, FLEET_SIZES):
        res = payload["station_results"][str(fleet)]
        for m in ("DaaS-GC", "DaaS-HGC", "DaaS-SG", "DaaS-KGE"):
            ax.plot(payload["station_sizes"], res[m], marker="o", label=m)
        ax.set_title(f"{fleet} drones")
        ax.set_xlabel("Number of stations")
    axes[0].set_ylabel("Computation time (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Computation time vs. sky-network size")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp4_time_vs_stations.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for fleet in FLEET_SIZES:
        res = payload["station_results"][str(fleet)]
        ax.plot(payload["station_sizes"], res["DaaS-GC"], marker="d", label=f"{fleet} drones")
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("DaaS-GC computation time (s)")
    ax.set_title("DaaS-GC scalability across fleet sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp4_gc_scalability.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2)[:2000])
