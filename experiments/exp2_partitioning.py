"""Experiment 2 - Impact of partitioning (DaaS-GC.tex Section 7.3.2).

Number of clusters k swept from 2 to 10 (via the (eps, mu) calibration
routine), across sky networks of 50, 100 and 200 stations. Records
delivery distance and number of traversed stations for DaaS-GC and
DaaS-HGC, with DaaS-SG/DaaS-KGE reported as fixed (non-clustered)
reference points.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daasgc.runner import run_daas_gc, run_daas_hgc
from baselines.daas_kge.run import run_daas_kge
from baselines.daas_sg import run_daas_sg
from experiments.common import FIGURES_DIR, RESULTS_DIR, load_instance

K_VALUES = list(range(2, 11))
STATION_SIZES = [50, 100, 200]


def _path_distance(sky, stations) -> float:
    if len(stations) < 2:
        return 0.0
    return sum(sky.segment_distance(stations[i], stations[i + 1]) for i in range(len(stations) - 1))


def run() -> dict:
    payload = {"k_values": K_VALUES, "station_sizes": STATION_SIZES, "results": {}}

    for n_stations in STATION_SIZES:
        sky, requests, catalog = load_instance(n_stations, 20)
        req = requests[0]
        src, tgt, items, weights = req["source"], req["target"], req["items"], req["weights"]

        gc_dist, gc_nstat, hgc_dist, hgc_nstat = [], [], [], []
        for k in K_VALUES:
            r_gc = run_daas_gc(sky, src, tgt, items, weights, target_k=k)
            gc_dist.append(round(_path_distance(sky, r_gc.path.stations), 3))
            gc_nstat.append(len(r_gc.path.stations))

            r_hgc = run_daas_hgc(sky, src, tgt, items, weights, target_k=k)
            hgc_dist.append(round(_path_distance(sky, r_hgc.path.stations), 3))
            hgc_nstat.append(len(r_hgc.path.stations))

        r_sg = run_daas_sg(sky, src, tgt, items, weights)
        r_kge = run_daas_kge(sky, catalog, src, tgt, items, weights, num_walks=3, walk_length=3, dimension=32)

        payload["results"][str(n_stations)] = {
            "DaaS-GC": {"distance": gc_dist, "n_stations": gc_nstat},
            "DaaS-HGC": {"distance": hgc_dist, "n_stations": hgc_nstat},
            "DaaS-SG": {"distance": round(_path_distance(sky, r_sg.path.stations), 3),
                        "n_stations": len(r_sg.path.stations)},
            "DaaS-KGE": {"distance": round(_path_distance(sky, r_kge.path.stations), 3),
                         "n_stations": len(r_kge.path.stations)},
        }

    with open(os.path.join(RESULTS_DIR, "exp2_partitioning.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _plot(payload)
    return payload


def _plot(payload: dict) -> None:
    k_values = payload["k_values"]

    fig, axes = plt.subplots(1, len(STATION_SIZES), figsize=(5 * len(STATION_SIZES), 4.5), sharey=True)
    for ax, n_stations in zip(axes, STATION_SIZES):
        res = payload["results"][str(n_stations)]
        ax.plot(k_values, res["DaaS-GC"]["distance"], marker="o", label="DaaS-GC")
        ax.plot(k_values, res["DaaS-HGC"]["distance"], marker="s", label="DaaS-HGC")
        ax.axhline(res["DaaS-SG"]["distance"], linestyle="--", color="gray", label="DaaS-SG")
        ax.axhline(res["DaaS-KGE"]["distance"], linestyle=":", color="black", label="DaaS-KGE")
        ax.set_title(f"{n_stations} stations")
        ax.set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Delivery distance")
    axes[0].legend(fontsize=8)
    fig.suptitle("Impact of partitioning on delivery distance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp2_distance.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(STATION_SIZES), figsize=(5 * len(STATION_SIZES), 4.5), sharey=True)
    for ax, n_stations in zip(axes, STATION_SIZES):
        res = payload["results"][str(n_stations)]
        ax.plot(k_values, res["DaaS-GC"]["n_stations"], marker="o", label="DaaS-GC")
        ax.plot(k_values, res["DaaS-HGC"]["n_stations"], marker="s", label="DaaS-HGC")
        ax.set_title(f"{n_stations} stations")
        ax.set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Number of traversed stations")
    axes[0].legend(fontsize=8)
    fig.suptitle("Impact of partitioning on the number of traversed stations")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp2_nstations.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n_stations in STATION_SIZES:
        res = payload["results"][str(n_stations)]
        gap = [g - h for g, h in zip(res["DaaS-GC"]["distance"], res["DaaS-HGC"]["distance"])]
        ax.plot(k_values, gap, marker="d", label=f"{n_stations} stations")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("DaaS-GC distance - DaaS-HGC distance")
    ax.set_title("DaaS-GC vs. DaaS-HGC distance gap across k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp2_gc_vs_hgc_gap.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2)[:2000])
