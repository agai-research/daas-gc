# DaaS-GC — Region-based Graph Clustering for UAV Delivery Service Composition

A runnable Python prototype of the DaaS-GC approach: a SCAN-inspired graph
clustering strategy that partitions a skyway network into sky regions, an
A\*-based delivery path search that exploits this partitioning, and a
drone-formation selection algorithm that composes the best set of drone
services for a delivery request.


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



## 3. Dataset

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
altitude, wind, route).

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

## 6. Execution environment

- Python 3.10+
- `networkx`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `gensim`

Install with:

```bash
pip install networkx numpy scipy pandas matplotlib scikit-learn gensim
```

All scripts were run and verified in this environment end to end
(`tests/first_test.py` and `main_exp.py` complete without error).
