"""Extraction of candidate drones and stations from the DaaSKG embedding
space (DaaS_KGE-1.pdf, Section 5.1 "Extraction of candidate drones and
stations"). Proximity to the delivery source/target stations is used to
restrict the search to a high-proximity vector subspace, avoiding a full
re-exploration of the sky network.
"""

from __future__ import annotations

from gensim.models import Word2Vec

from daasgc.sky_network import SkyNetwork
from baselines.daas_kge.embedding import cosine_proximity


def candidate_stations(model: Word2Vec, source: str, target: str, station_ids: list[str],
                        top_n: int = 20) -> list[str]:
    """High-proximity stations w.r.t. the source/target embeddings (the
    'nearby stations' olive markers of Fig. 2 in DaaS_KGE-1.pdf)."""
    scored = []
    for sid in station_ids:
        node = f"S::{sid}"
        prox = max(cosine_proximity(model, node, f"S::{source}"),
                   cosine_proximity(model, node, f"S::{target}"))
        scored.append((sid, prox))
    scored.sort(key=lambda t: t[1], reverse=True)
    top = {sid for sid, _ in scored[:top_n]}
    top.add(source)
    top.add(target)
    return list(top)


def candidate_drones(sky: SkyNetwork, model: Word2Vec, station_subset: list[str],
                      top_n: int = 15) -> list[str]:
    """High-proximity drones w.r.t. the candidate station subspace."""
    station_nodes = [f"S::{sid}" for sid in station_subset]
    scored = []
    for d in sky.drones:
        node = f"D::{d.id}"
        prox = max((cosine_proximity(model, node, sn) for sn in station_nodes), default=0.0)
        scored.append((d.id, prox))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [did for did, _ in scored[:top_n]]
