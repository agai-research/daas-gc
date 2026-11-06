"""Build every dataset instance used by the experiments.

Sizes: stations in {50, 100, 150, 200, 250, 300} x drones in {10, 20, 30}.
Each instance folder s<stations>_d<drones> is fully determined by
data/build_dataset.py (network seed = 10000 + stations, fleet seed = 20000).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.build_dataset import build_and_save  # noqa: E402

STATION_SIZES = [50, 100, 150, 200, 250, 300]
FLEET_SIZES = [10, 20, 30]


def build_all() -> list[str]:
    return [build_and_save(s, d) for s in STATION_SIZES for d in FLEET_SIZES]


if __name__ == "__main__":
    for p in build_all():
        print(p)
