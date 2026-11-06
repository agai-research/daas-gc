"""Draw the six figures of the experiments section from results/*.json,
in the visual style of the submitted manuscript.

    python plots/make_figures.py            # redraw every figure whose JSON exists

Bars are means over the runs, box plots show the distribution over the runs.
Experiments are run for every k = 1..10; Figs. 6, 7 and 9 show k = 1 (the
single-cluster case, identical to DaaS-SG; reviewer comment 2) and the even
values 2..10 used in the manuscript. DaaS-SG, which does not partition the
network, is the k = 1 case: Figs. 6 and 9 draw it in the k = 1 group (own bar,
distance marker and box), next to the identical DaaS-GC and DaaS-HGC values.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"],
    "hatch.linewidth": 0.8, "axes.linewidth": 1.0, "legend.framealpha": 0.9,
})
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402,F401
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

METHODS = ["DaaS-GC", "DaaS-HGC", "DaaS-KGE", "DaaS-SG"]
K_SHOWN = [1, 2, 4, 6, 8, 10]

# One texture per method, used in every figure: Okabe-Ito colour-blind-safe colour + hatch;
# `light` is the tint used for a second series of the same method (battery usage in Fig. 7).
STYLE = {
    "DaaS-GC":  {"color": "#0072B2", "light": "#9cc9e6", "hatch": "///",   "line": "#005c8f", "marker": "o"},
    "DaaS-HGC": {"color": "#E69F00", "light": "#f6d68a", "hatch": "...",   "line": "#bf8400", "marker": "x"},
    "DaaS-KGE": {"color": "#009E73", "light": "#99d8c7", "hatch": "xxx",   "line": "#00704f", "marker": "s"},
    "DaaS-SG":  {"color": "#CC79A7", "light": "#ebc9dc", "hatch": "\\\\\\", "line": "#a3517f", "marker": "D"},
}
EDGE_LW = 0.8
# white median with a black outline: visible on every fill colour and hatch
MEDIAN = {"color": "white", "linewidth": 2.0,
          "path_effects": [matplotlib.patheffects.Stroke(linewidth=3.4, foreground="black"),
                           matplotlib.patheffects.Normal()]}
GRID = {"linestyle": "--", "color": "#cccccc", "linewidth": 0.8}
DPI = 300


def _patch(m, light=False, label=None):
    st = STYLE[m]
    return Patch(facecolor=st["light"] if light else st["color"], edgecolor="black", hatch=st["hatch"],
                 linewidth=EDGE_LW, label=label or m)


def _bar(ax, x, h, w, m, light=False, bottom=None, label=None, zorder=2):
    st = STYLE[m]
    return ax.bar(x, h, w, bottom=bottom, color=st["light"] if light else st["color"], edgecolor="black",
                  linewidth=EDGE_LW, hatch=st["hatch"], label=label, zorder=zorder)


def _style_box(bp, m):
    for b in bp["boxes"]:
        b.set_facecolor(STYLE[m]["color"])
        b.set_edgecolor("black")
        b.set_linewidth(EDGE_LW)
        b.set_hatch(STYLE[m]["hatch"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def _vals(recs, key):
    return [r.get(key) for r in recs if not r.get("failed")]


def _ylim(ax, lo, hi, data_max):
    """Use the manuscript's axis range when the data fit in it, else extend it."""
    ax.set_ylim(lo, hi if data_max <= hi * 0.98 else data_max * 1.25)


def _save(fig, name, dpi=DPI):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Fig. 5
def fig_drone_cost_time(p):
    res, payloads = p["results"], p["config"]["payloads_g"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    width = 150
    specs = (("n_drones", "# Drones", 6, "upper left"), ("delivery_cost", "Cost", 250, "upper left"),
             ("delivery_time_min", "Delivery time (min)", 200, "upper left"))
    for ax, (key, ylabel, ymax, loc) in zip(axes, specs):
        top = 0
        for i, m in enumerate(METHODS):
            means = [_mean(_vals(res[m][str(w)], key)) for w in payloads]
            top = max(top, max(means))
            _bar(ax, [w + (i - 1.5) * width for w in payloads], means, width, m, label=m)
        ax.set_xticks(payloads)
        ax.set_xlabel("Payload (g)")
        ax.set_ylabel(ylabel)
        _ylim(ax, 0, ymax, top)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax.legend(loc=loc)
    fig.tight_layout()
    return _save(fig, "drone-cost-time.png")


# ---------------------------------------------------------------- Fig. 6
K1_LABEL = "1\n(= DaaS-SG)"


def _k_labels():
    return [K1_LABEL if k == 1 else str(k) for k in K_SHOWN]


def fig_stations_distance(p):
    """DaaS-SG does not partition the network, so it has a single value per network size; it is
    the k = 1 case and is drawn there (own bar and distance marker), not as a line across k."""
    sizes = p["config"]["stations"]
    fig, axes = plt.subplots(1, len(sizes), figsize=(18, 6))
    w = 0.25
    for idx, (ax, n) in enumerate(zip(axes, sizes)):
        res = p["sizes"][str(n)]["results"]
        ax2 = ax.twinx()
        ax2.set_zorder(ax.get_zorder() + 1)      # distance lines above the bars
        ax2.patch.set_visible(False)
        x = list(range(len(K_SHOWN)))
        nmax = 0
        for j, k in enumerate(K_SHOWN):
            methods = ["DaaS-GC", "DaaS-HGC"] + (["DaaS-SG"] if k == 1 else [])
            for i, m in enumerate(methods):
                recs = res[m] if m == "DaaS-SG" else res[m][str(k)]
                v = _mean(_vals(recs, "n_stations"))
                nmax = max(nmax, v)
                _bar(ax, j + (i - (len(methods) - 1) / 2) * w, v, w, m,
                     label=f"{m} (#Stations)" if j == 0 else None)
        gc_d = [_mean(_vals(res["DaaS-GC"][str(k)], "distance")) for k in K_SHOWN]
        hgc_d = [_mean(_vals(res["DaaS-HGC"][str(k)], "distance")) for k in K_SHOWN]
        sg_d = _mean(_vals(res["DaaS-SG"], "distance"))
        for m, d in (("DaaS-GC", gc_d), ("DaaS-HGC", hgc_d)):
            ax2.plot(x, d, ls="--", lw=1.6, color=STYLE[m]["line"], marker=STYLE[m]["marker"], ms=7,
                     mfc="white" if m == "DaaS-GC" else STYLE[m]["line"], mec=STYLE[m]["line"], mew=1.6,
                     label=f"{m} (Distance)")
        ax2.plot([0], [sg_d], "D", ms=12, mfc="none", mec=STYLE["DaaS-SG"]["line"], mew=2,
                 label="DaaS-SG (Distance)")
        ax.set_xticks(x, _k_labels())
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("#Stations")
        ax2.set_ylabel("Distance (km)")
        ax.set_title(f"{n} stations")
        _ylim(ax, 1, 10, nmax)
        dmax, dmin = max(gc_d + hgc_d + [sg_d]), min(gc_d + hgc_d + [sg_d])
        ax2.set_ylim(50 if dmin >= 52 else max(0, dmin * 0.8), 150 if dmax <= 147 else dmax * 1.08)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax2.add_artist(ax2.legend(handles=ax.get_legend_handles_labels()[0], loc="upper left"))
        if idx == len(sizes) - 1:
            ax2.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "stations-distance.png")


# ---------------------------------------------------------------- Fig. 7
def fig_energy(p):
    e = p["energy"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(K_SHOWN)))
    w = 0.15
    top = 0
    for i, m in enumerate(("DaaS-GC", "DaaS-HGC")):
        pos = [xi + (i - 0.5) * w for xi in x]
        eff = [_mean(_vals(e[m][str(k)], "energy_efficiency_pct")) for k in K_SHOWN]
        bat = [_mean(_vals(e[m][str(k)], "battery_usage_pct")) for k in K_SHOWN]
        top = max(top, max(a + b for a, b in zip(eff, bat)))
        # energy efficiency (full colour) with battery usage stacked on top (tint, same hatch);
        # the height of the tinted segment is the battery usage
        _bar(ax, pos, eff, w, m, label=f"{m} (Energy efficiency)", zorder=3)
        _bar(ax, pos, bat, w, m, light=True, bottom=eff, label=f"{m} (Battery usage)", zorder=3)
    ax.set_xticks(x, _k_labels())
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Percentage (%)")
    _ylim(ax, 0, 120, top)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center")
    fig.tight_layout()
    return _save(fig, "energy.png")


# ---------------------------------------------------------------- box plots
def _boxes(ax, groups, xlabels, methods, width, fill=0.75):
    """Grouped box plots; `width` = distance between box centres, `fill` = box width / width
    (fractions of the spacing between x groups)."""
    n = len(methods)
    for i, m in enumerate(methods):
        pos = [j + (i - (n - 1) / 2) * width for j in range(len(xlabels))]
        bp = ax.boxplot([groups[m][j] for j in range(len(xlabels))], positions=pos, widths=width * fill,
                        patch_artist=True, showfliers=False, manage_ticks=False, medianprops=MEDIAN)
        _style_box(bp, m)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_xlim(-0.6, len(xlabels) - 0.4)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)


def _whisker(v, upper=True):
    """End of the box-plot whisker (1.5 IQR rule), i.e. what the figure shows."""
    q1, _, q3 = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else (v[0], v[0], v[0])
    iqr = q3 - q1
    if upper:
        return max(x for x in v if x <= q3 + 1.5 * iqr)
    return min(x for x in v if x >= q1 - 1.5 * iqr)


def _data_max(groups):
    return max(_whisker(v) for g in groups.values() for v in g)


def _data_min(groups):
    return min(_whisker(v, upper=False) for g in groups.values() for v in g)


# ---------------------------------------------------------------- Fig. 8
def fig_failure(p):
    f, payloads = p["failure"], p["config"]["payloads_g_failure"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    groups = {m: [[r["failure_rate_pct"] for r in f[m][str(w)]] for w in payloads] for m in METHODS}
    _boxes(ax, groups, payloads, METHODS, 0.19, fill=0.8)
    ax.set_xlabel("Payload (g)")
    ax.set_ylabel("Failure rate (%)")
    lo, hi = _data_min(groups), _data_max(groups)
    ax.set_ylim(15 if lo >= 15 else max(0, lo - 3), 60 if hi <= 58 else hi * 1.08)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.legend(handles=[_patch(m) for m in METHODS], loc="upper right")
    fig.tight_layout()
    return _save(fig, "failure-boxplots.png")


# ---------------------------------------------------------------- Fig. 9
def fig_comp_time_clusters(p):
    """DaaS-SG is drawn as its own box in the k = 1 group (its search is the k = 1 search)."""
    sizes = p["config"]["sizes_a"]
    fig, axes = plt.subplots(1, len(sizes), figsize=(16.2, 5.3))
    step, fill = 0.28, 0.8
    for ax, n in zip(axes, sizes):
        res = p["time_vs_k_ms"][str(n)]
        allv = []
        for j, k in enumerate(K_SHOWN):
            methods = ["DaaS-GC", "DaaS-HGC"] + (["DaaS-SG"] if k == 1 else [])
            for i, m in enumerate(methods):
                v = res[m] if m == "DaaS-SG" else res[m][str(k)]
                allv.append(v)
                bp = ax.boxplot([v], positions=[j + (i - (len(methods) - 1) / 2) * step], widths=step * fill,
                                patch_artist=True, showfliers=False, manage_ticks=False,
                                medianprops=MEDIAN)
                _style_box(bp, m)
        ax.set_xticks(range(len(K_SHOWN)), _k_labels())
        ax.set_xlim(-0.6, len(K_SHOWN) - 0.4)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        lo, hi = min(_whisker(v, upper=False) for v in allv), max(_whisker(v) for v in allv)
        ax.set_ylim(max(0, lo - 0.08 * (hi - lo)), hi + 0.1 * (hi - lo))
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Computation time (ms)")
        ax.set_title(f"{n} stations")
        ax.legend(handles=[_patch(m) for m in ("DaaS-GC", "DaaS-HGC", "DaaS-SG")], loc="upper right")
    fig.tight_layout()
    return _save(fig, "comp-time-clusters-boxplots.png")


# ---------------------------------------------------------------- Fig. 10
def fig_comp_time_sn(p):
    fleets, sizes = p["config"]["fleets_b"], p["config"]["sizes_b"]
    fig, axes = plt.subplots(1, len(fleets), figsize=(20, 6))
    for idx, (ax, d) in enumerate(zip(axes, fleets)):
        res = p["time_vs_stations_ms"][str(d)]
        groups = {m: [res[m][str(n)] for n in sizes] for m in METHODS}
        _boxes(ax, groups, sizes, METHODS, 0.2, fill=0.8)
        lo, hi = _data_min(groups), _data_max(groups)
        ax.set_ylim(max(0, lo - 0.05 * (hi - lo)), hi + 0.08 * (hi - lo))
        ax.set_xlabel("Number of stations")
        ax.set_ylabel("Computation time (ms)")
        ax.set_title(f"{d} drones")
        ax.legend(handles=[_patch(m) for m in METHODS],
                  loc="upper right" if idx == len(fleets) - 1 else "upper left")
    fig.tight_layout()
    return _save(fig, "comp-time-sn-boxplots.png")


def main():
    jobs = [("exp1_user_request", [fig_drone_cost_time]), ("exp2_partitioning", [fig_stations_distance]),
            ("exp3_energy_failure", [fig_energy, fig_failure]),
            ("exp4_computation_time", [fig_comp_time_clusters, fig_comp_time_sn])]
    for name, fns in jobs:
        path = os.path.join(RESULTS, f"{name}.json")
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found", file=sys.stderr)
            continue
        with open(path) as f:
            payload = json.load(f)
        for fn in fns:
            print(fn(payload))


if __name__ == "__main__":
    main()
