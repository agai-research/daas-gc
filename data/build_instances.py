"""Build every dataset instance needed by the experiment scripts.

Instance naming: s<stations>_d<drones> under data/instances/. Sized to
match the sweeps used in Section 7.3 of DaaS-GC.tex:
- Exp 1 (impact of user request): one mid-size instance, k fixed to 5.
- Exp 2 (impact of partitioning): stations in {50, 100, 200}.
- Exp 3 (energy/failure): stations=100 instance, k in [2,10] and payload sweep.
- Exp 4 (computational efficiency): stations in [50,300] step 50, drones in {10,20,30}.
"""

from __future__ import annotations

import os

from data.build_dataset import build_and_save

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def build_all() -> list[str]:
    built = []

    # Exp 1 & Exp 3: a single reference instance (100 stations, 20 drones).
    built.append(build_and_save(n_stations=100, n_drones=20, seed=42))

    # Exp 2: partitioning sweep across sky-network sizes.
    for n_stations in (50, 100, 200):
        built.append(build_and_save(n_stations=n_stations, n_drones=20, seed=100 + n_stations))

    # Exp 4: computational-efficiency sweep across sizes and fleet sizes.
    for n_stations in (50, 100, 150, 200, 250, 300):
        for n_drones in (10, 20, 30):
            built.append(build_and_save(n_stations=n_stations, n_drones=n_drones,
                                         seed=1000 + n_stations + n_drones))

    return built


if __name__ == "__main__":
    paths = build_all()
    for p in paths:
        print(p)
    print(f"Built {len(paths)} dataset instances.")
