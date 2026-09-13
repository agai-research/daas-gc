# DaaS-GC — Region-based Graph Clustering for UAV Delivery Service Composition

A runnable Python prototype of the DaaS-GC approach: a SCAN-inspired graph
clustering strategy that partitions a skyway network into sky regions, an
A\*-based delivery path search that exploits this partitioning, and a
drone-formation selection algorithm that composes the best set of drone
services for a delivery request. The prototype also implements three
baselines used in the paper's experimental validation: **DaaS-HGC**
(hyper-graph variant), **DaaS-SG** (single, unpartitioned graph), and
**DaaS-KGE** (knowledge-graph-embedding composition).

![DaaS-GC pipeline](docs/approach_figure.png)

## 1. Repository layout

```
DaaS-GC/
  daasgc/               core prototype: sky network, clustering, path search, composition
    sky_network.py       graph + drone/request data structures, JSON/GraphML loading
    clustering.py         Algorithm 1 - SCAN-inspired partitioning + (eps, mu) calibration
    hypergraph.py          DaaS-HGC hyper-graph-first path search
    path_search.py         Algorithm 2 - A* delivery path search (3 scenarios)
    composition.py          Algorithm 3 - drone formation selection + QoS scoring
    runner.py                 high-level run_daas_gc() / run_daas_hgc()
  baselines/
    daas_sg.py             single-graph baseline (exact A*, k=1)
    daas_kge/                knowledge-graph-embedding baseline
      kg_build.py            DaaSKG construction (typed entities/relations)
      embedding.py             metapath2vec-style guided walks + Skip-Gram
      proximity.py              candidate drone/station extraction
      composition.py             proximity-aware formation selection (SCDaaS)
      path_selection.py          Dijkstra-based skyway path selection
      run.py                     end-to-end DaaS-KGE pipeline
  data/
    build_dataset.py       builds one dataset instance (stations, drones, requests)
    build_instances.py       builds every instance needed by the experiments
    instances/                  generated JSON dataset instances (git-ignoreable)
  experiments/            one isolated script per experiment series (Section 7.3)
  results/                 JSON results per experiment
  figures/                  PNG figures per experiment (>= 3 per experiment)
  notebooks/               Colab-ready .ipynb notebooks
  tests/                  first_test.py - end-to-end smoke test
  docs/                    this README, Results.tex, dataset/results summary
  daasgc.py               CLI entry point: run one method on one request
  main_exp.py              builds all instances and runs all experiments
```

## 2. Approach summary

| Step | Module | Paper reference |
|---|---|---|
| Sky network modeling | `daasgc/sky_network.py` | Section "Proposed Approach" |
| Structural clustering (core/border/frontier/bridge/outlier) | `daasgc/clustering.py` | Algorithm 1, Eq. eq:similarity / eq:strongconnection / eq:core |
| Hyper-graph abstraction | `daasgc/clustering.py::build_hypergraph`, `daasgc/hypergraph.py` | Definition "Sky network hyper graph" |
| Delivery path search (single/two/multi-cluster) | `daasgc/path_search.py` | Algorithm 2, Eq. eq:path-score |
| Drone formation selection | `daasgc/composition.py` | Algorithm 3, Eq. eq:daas-score |
| DaaS-KGE baseline | `baselines/daas_kge/` | DaaS_KGE-1.pdf, Sections 4-5 |

Clustering parameters `eps` (similarity threshold) and `mu` (core-station
degree threshold) are never fixed inside the code: `clustering.calibrate_for_target_k`
searches a grid to approximate a requested number of clusters `k`, matching
the calibration procedure used in the "Impact of partitioning" experiment.
QoS weights (`w1..w4`) come from the delivery request, not from a hardcoded
constant, so different request profiles (emergency, commercial,
trust-sensitive, balanced) can be tested directly.

## 3. Dataset

Built entirely from `data/build_dataset.py`, following Table "Dataset and
variable settings" in DaaS-GC.tex:

| Variable | Value |
|---|---|
| # drones | [10, 30] |
| # recharging stations | [50, 300] |
| # recharging pads | [200, 1200] |
| # skyway segments | [245, 8970] |
| Full recharging time | ~60 minutes |
| Drone speed | 65 km/h |
| Max weight per package | 5 kg |
| # items per request | 8 |

The drone flight metadata schema mirrors the public CMU KiLTHub "Data
Collected with Package Delivery Quadcopter Drone" dataset (30 outdoor
drones: battery voltage/current, position, orientation, speed, payload,
altitude, wind, route). Since this environment has no internet access to
KiLTHub, a same-schema synthetic stand-in is constructed instead — see
`implementation_report.docx` for details. The skyway topology (stations,
pads, segments) has no equivalent public dataset at the required scale
and is constructed synthetically, exactly as described in the paper's own
Section 7.1.

## 4. Running the prototype

```bash
cd DaaS-GC
pip install -r requirements.txt

# Build one dataset instance (100 stations, 20 drones)
python -c "from data.build_dataset import build_and_save; build_and_save(100, 20)"

# Run a smoke test on a small, self-contained sample sky network
python tests/first_test.py

# Run DaaS-GC on the first request of that instance
python daasgc.py --instance data/instances/s100_d20 --method DaaS-GC --k 5

# Run any baseline the same way
python daasgc.py --instance data/instances/s100_d20 --method DaaS-HGC --k 5
python daasgc.py --instance data/instances/s100_d20 --method DaaS-SG
python daasgc.py --instance data/instances/s100_d20 --method DaaS-KGE
```

## 5. Reproducing the experiments

```bash
python main_exp.py
```

This builds every dataset instance needed (`data/build_instances.py`) and
runs the four experiment series from Section 7.3 of DaaS-GC.tex, saving:
- JSON results under `results/`
- at least three figures per experiment under `figures/`

| Script | Experiment |
|---|---|
| `experiments/exp1_user_request.py` | Impact of user request (package weight) |
| `experiments/exp2_partitioning.py` | Impact of partitioning (number of clusters k) |
| `experiments/exp3_energy_failure.py` | Energy efficiency and failure rate |
| `experiments/exp4_computation_time.py` | Computational efficiency |

## 6. Execution environment

- Python 3.10+
- `networkx`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `gensim`

Install with:

```bash
pip install networkx numpy scipy pandas matplotlib scikit-learn gensim
```

All scripts were run and verified in this environment end to end
(`tests/first_test.py` and `main_exp.py` complete without error).

## 7. Notebooks

`notebooks/` contains Colab-ready notebooks:
- `01_run_all_methods.ipynb` — builds a dataset instance, runs all four
  methods on a chosen request, shows a results table and inline figures.
- `02_data_and_results_report.ipynb` — narrative report over the dataset
  and the experiment results, with explanatory text cells.
- `03_daas_gc_hgc.ipynb`, `04_daas_sg.ipynb`, `05_daas_kge.ipynb` —
  one notebook per method, demonstrating it in isolation.

## 8. Notes and limitations

Any adjustment made to an unrealistic algorithmic step (e.g., translating
a `GOTO` in the paper's pseudocode into an equivalent loop), or any
dataset substitution, is documented in `implementation_report.docx`, kept
separate from this README so the latter stays a clean, reusable
description of the prototype.
