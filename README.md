# DaaS-GC — Region-based Graph Clustering for UAV Delivery Service Composition

A runnable Python prototype of the DaaS-GC approach: a SCAN-inspired graph
clustering strategy that partitions a skyway network into sky regions, an
A\*-based delivery path search that exploits this partitioning, and a
drone-formation selection algorithm that composes the best set of drone
services for a delivery request.

**Article:** Partitioning the Sky: A Region-based Graph Clustering Approach for the Composition of UAV Delivery Services
**Authors:** Hela Elmannai, Haithem Mezni, Abeer D. Algarni


## 1. Dataset

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

## 2. Running the prototype

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

## 3. Execution environment

- Python 3.10+
- `networkx`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `gensim`

Install with:

```bash
pip install networkx numpy scipy pandas matplotlib scikit-learn gensim
```

All scripts were run and verified in this environment end to end
(`tests/first_test.py` and `main_exp.py` complete without error).
