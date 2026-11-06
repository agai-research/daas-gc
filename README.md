# DaaS-GC — Region-based Graph Clustering for UAV Delivery Service Composition

Implementation and experiments of the article *Partitioning the Sky: A Region-based Graph
Clustering Approach for the Composition of UAV Delivery Services* (H. Elmannai, H. Mezni,
A. D. Algarni): SCAN-based partitioning of a skyway network (Algorithm 1), A*-based delivery
path search over the partitions (Algorithm 2, DaaS-GC; hyper-graph variant DaaS-HGC), drone
formation selection (Algorithm 3), and the baselines DaaS-SG (exact A* on the whole network)
and DaaS-KGE (knowledge-graph embedding).

## 1. Reproduce every figure (one command)

```bash
git clone https://github.com/agai-research/daas-gc.git
cd daas-gc
pip install -r requirements.txt          # Python >= 3.10; versions used: see requirements.txt
python main_exp.py --runs 10 --seed 42   # ~40 s on a laptop
```

This builds the 18 dataset instances (`data/instances/`, also committed), runs the four
experiments, writes `results/*.json` and draws the six figures in `figures/` under the file
names used in the article:

| Article figure | File | Produced by |
|---|---|---|
| Fig. 5 — impact of delivery requests | `figures/drone-cost-time.png` | `experiments/exp1_user_request.py` |
| Fig. 6 — impact of partitioning (k = 1…10; DaaS-SG shown at k = 1) | `figures/stations-distance.png` | `experiments/exp2_partitioning.py` |
| Fig. 7 — energy efficiency | `figures/energy.png` | `experiments/exp3_energy_failure.py` |
| Fig. 8 — failure rate | `figures/failure-boxplots.png` | `experiments/exp3_energy_failure.py` |
| Fig. 9 — computation time vs k | `figures/comp-time-clusters-boxplots.png` | `experiments/exp4_computation_time.py` |
| Fig. 10 — computation time vs network size | `figures/comp-time-sn-boxplots.png` | `experiments/exp4_computation_time.py` |

All drawing code is in `plots/make_figures.py`; `python plots/make_figures.py` redraws the
figures from the committed `results/*.json` without re-running anything.

Each experiment can also be run alone, with the same options:

```bash
python experiments/exp1_user_request.py --runs 10 --seed 42
python experiments/exp4_computation_time.py --runs 10 --seed 42
```

Figures use one texture per method (colour-blind-safe colour + hatch), so they stay readable
in greyscale print: DaaS-GC blue `///`, DaaS-HGC orange `...`, DaaS-KGE green `xxx`, DaaS-SG
purple `\\\`; 300 dpi.

## 2. Experimental protocol

* **Run** = one random delivery request: a source/target pair not directly linked by a segment
  and 8 items splitting the package weight; QoS weights w1 = w2 = w3 = w4 = 0.25. Requests come
  from generators seeded with (`--seed`, experiment, run, …), so `--seed 42` always gives the same
  requests. Bars and lines show means over `--runs` requests (per-run values are in `results/*.json`), box
  plots the distribution. Figs. 6, 7 and 9 show k = 1, 2, 4, 6, 8, 10; every k = 1…10 is in the
  JSON. In Figs. 6 and 9 the k = 1 group, labelled "1 (= DaaS-SG)", also shows DaaS-SG itself.
* **Pre-processing is never timed**, for any method: (ε, μ) calibration, clustering, regions and
  hyper-graph for DaaS-GC/HGC; knowledge-graph build and embedding training for DaaS-KGE. Their cost
  is reported separately in `results/exp4_computation_time.json` (`preprocessing_ms`).
* **Computation time** = query only (path search + formation selection), median of 3 executions,
  in ms. It depends on the machine; trends and rankings are what is comparable.
* Experiments 2–4 compute every k = 1…10 (all stored in `results/*.json`); Figs. 6, 7 and 9 show
  k = 1 and the even values 2…10 used in the article.
* **k = 1** is the single-cluster partition: DaaS-GC and DaaS-HGC then run exactly the DaaS-SG
  search. `exp2_partitioning.py` checks this for every request and stops if it is ever violated;
  `tests/test_k1_consistency.py` tests it independently.

| Experiment | Stations | Drones | k | Payload |
|---|---|---|---|---|
| exp1 (Fig. 5) | 100 | 20 | 5 | 1000…5000 g |
| exp2 (Fig. 6) | 50, 100, 200 | 20 | 1…10 | 3000 g |
| exp3 (Fig. 7) | 100 | 20 | 1…10 | 3000 g |
| exp3 (Fig. 8) | 100 | 20 | 5 | 1000…5000 g, 200 Monte-Carlo trials per plan |
| exp4 (Fig. 9) | 50, 100, 200 | 20 | 1…10 | 3000 g |
| exp4 (Fig. 10) | 50…300 | 10, 20, 30 | 5 | 3000 g |

## 3. Dataset

`data/build_dataset.py` (deterministic: network seed = 10000 + #stations, fleet seed = 20000;
the d-drone fleet is the first d drones of one master fleet of 30).

| Variable | Value |
|---|---|
| # drones | 10, 20, 30 |
| # recharging stations | 50 … 300 |
| # recharging pads | 2–6 per station (mean 4) → ≈ 200 … 1 200 |
| # skyway segments | 0.2 · n(n−1)/2 closest station pairs → 245 … 8 970 |
| Airspace | 100 km × 100 km, stations placed uniformly; segment distance = Euclidean distance |
| Full recharging time | 60 min (scaled by restored energy and charging voltage: 12 V ×1.25, 24 V ×1, 48 V ×0.8) |
| Drone speed | 65 km/h |
| Max weight per package / items per request | 5 kg / 8 |

**Data provenance.** The drone attributes follow the schema of the public CMU KiLTHub dataset
*Data Collected with Package Delivery Quadcopter Drone*, but that file is not redistributed here;
the fleet is a synthetic stand-in drawn from the distributions in `build_fleet()` (payload
0.8–3 kg, empty mass 2–6 kg, battery 200–500 Wh, 4–7 Wh/km empty, two supported pad types out of
three, cost, availability and reputation).

## 4. Delivery model (shared by all methods, `daasgc/delivery_model.py`)

* energy per km grows linearly with total mass; planning keeps a 20 % battery reserve;
* a drone recharges (to full) at an intermediate station only if it cannot otherwise reach its next
  charging opportunity, and only on a compatible pad type;
* the formation flies as a convoy; drones that charge at the same station share its free pads
  (sequential charging when there are more drones than free pads);
* delivery cost = Σ [drone service price (cheapest offer in `daas_catalog.json`) + cost per km ×
  distance] + charging-session fees ("aggregate cost of the selected drone services and the
  recharging expenses"); delivery time = flight time + waiting and charging time at stations;
* energy efficiency = energy needed on the straight source–target line / energy used; battery usage
  = battery depletion, i.e. deepest discharge between two full charges as a % of capacity (Fig. 7
  stacks it on top of energy efficiency, as in the submitted figure); the total energy drawn over the
  trip is stored separately (`battery_energy_drawn_pct`);
* failure rate = share of Monte-Carlo trials where a drone fails (1 − availability), runs out of
  energy under wind (0–12 m/s, +2 % consumption per m/s), or the delivery exceeds 1.2 × the planned
  time (pad occupancy drawn from each station's congestion, waiting time uniform on [0, 2 × average]).

## 5. Code map

| Paper | Code |
|---|---|
| Eq. similarity, ε-neighbourhood, core/border/bridge/outlier, Algorithm 1 (SCAN) with the functional criterion of Def. Cluster (strong connection = σ ≥ ε **and** same pad type) | `daasgc/clustering.py` (`cluster_sky_network`, `functionally_similar`, `FUNCTIONAL_GATE`) |
| Calibration of (ε, μ) for a target k; k = 1 | `daasgc/clustering.py` (`calibration_sweep`, `calibrate_for_target_k`, `single_cluster`) |
| Sky-network hyper-graph | `daasgc/clustering.py` (`build_hypergraph`) |
| Eq. path-score, A*, Algorithm 2 (DaaS-GC: every entry × every exit gateway of each traversed cluster, cheapest chain by dynamic programming) | `daasgc/path_search.py` (`delivery_path_search`) |
| Algorithm 2 pseudo-code reading (greedy exit choice from one entry) | `daasgc/path_search.py`, set `GC_ALL_ENTRIES = False` |
| DaaS-HGC | `daasgc/hypergraph.py` |
| Algorithm 3, Eq. daas-score | `daasgc/composition.py` |
| Pre-processing / query split | `daasgc/runner.py` |
| DaaS-SG | `baselines/daas_sg.py` |
| DaaS-KGE | `baselines/daas_kge/` |

Note on frontier stations: the paper's definition (clustered station with a cross-cluster edge
with σ ≥ ε) is computed and reported (`frontier_paper_def`), but SCAN separates clusters along
weak edges, so this set is almost always empty; path search therefore uses *gateway* stations
(clustered stations with any edge leaving their cluster).

## 6. Single request, tests

```bash
python daasgc.py --stations 100 --drones 20 --method DaaS-GC --k 5 --payload 3000 --seed 1
python tests/first_test.py                 # smoke test
python tests/test_k1_consistency.py        # k = 1 ≡ DaaS-SG, SCAN invariant
```

All results except computation times are bit-identical across runs, machines' hash seeds and
processes (gensim uses a deterministic hash function and a single worker). After a run, `git status`
lists `results/` and `figures/` as modified: the JSON files also record each query's computation
time, and PNG rendering (font anti-aliasing) can differ slightly between systems. All other values are
unchanged; `git checkout -- results figures` restores the committed versions.

See `CHANGELOG.md` for the differences with the version first shared with the reviewers.
