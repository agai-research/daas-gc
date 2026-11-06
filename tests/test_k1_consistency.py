"""k = 1 must coincide with DaaS-SG (reviewer comment 2).

With a single cluster, DaaS-GC and DaaS-HGC run the same exact A* over the
whole sky network as DaaS-SG, so path, formation, cost and time must be
identical. Run:  python tests/test_k1_consistency.py   (or pytest).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import load_instance, make_items, pick_pair, query, record, rng_for, same_plan  # noqa: E402


def test_k1_equals_daas_sg(sizes=(50, 100, 200), requests=20):
    for n in sizes:
        inst = load_instance(n, 20)
        for r in range(requests):
            rng = rng_for(0, "k1-test", n, r)
            s, t = pick_pair(inst.sky, rng)
            items = make_items(rng.choice([1000, 3000, 5000]), rng)
            sg = record(inst, query(inst, "DaaS-SG", s, t, items, repeats=1)[0])
            for m in ("DaaS-GC", "DaaS-HGC"):
                rec = record(inst, query(inst, m, s, t, items, k=1, repeats=1)[0])
                assert same_plan(rec, sg), f"{m} k=1 != DaaS-SG on s{n}_d20, request {r}"


def test_partition_is_scan():
    """Strongly-connected core stations are never split across clusters, and every
    cluster is functionally homogeneous (single pad type)."""
    from daasgc.clustering import functionally_similar, partition, similarity_table
    inst = load_instance(100, 20)
    sim = similarity_table(inst.sky)
    for k in range(2, 11):
        c = partition(inst.sky, k)
        for u in c.core:
            for v in inst.sky.graph.neighbors(u):
                if v in c.core and sim[u][v] >= c.eps and functionally_similar(inst.sky, u, v):
                    assert c.cluster_of(u) == c.cluster_of(v), (k, u, v)
        for members in c.clusters.values():
            assert len({inst.sky.functional_profile(m) for m in members}) == 1, k


if __name__ == "__main__":
    test_k1_equals_daas_sg()
    test_partition_is_scan()
    print("OK: k = 1 coincides with DaaS-SG; SCAN clusters never split strongly-connected cores and are functionally homogeneous")
