"""Metapath2vec embedding of the DaaSKG (DaaS_KGE-1.pdf, Sections 5.1.2/5.1.3).

Guided random walks follow a predefined meta-path of typed nodes; the
resulting node sequences are fed to Skip-Gram (gensim Word2Vec, sg=1) to
learn low-dimensional embeddings. Meta-paths default to the ones named in
DaaS-GC.tex Section 7 ("we used [S]+, [D]+ and D-S-D as the guiding
meta-paths"); walk count, walk length, window size and embedding
dimension are all configurable (never hardcoded), matching the
sensitivity study of DaaS_KGE-1.pdf (walks in [1,5], length in [2,4],
window in [2,5], dimension in [64,512]).
"""

from __future__ import annotations

import random

import networkx as nx
from gensim.models import Word2Vec

KIND_OF = {"station": "S", "drone": "D", "pad": "P", "daas": "A"}

DEFAULT_METAPATHS = ["S+", "D+", "DSD"]  # [S]+, [D]+, D-S-D


def _kind(kg: nx.MultiDiGraph, node: str) -> str:
    return KIND_OF.get(kg.nodes[node].get("kind"), "?")


def _next_step(kg: nx.MultiDiGraph, node: str, expected_kind: str | None, rng: random.Random) -> str | None:
    """One guided random-walk step: uniformly pick a neighbor, optionally
    constrained to `expected_kind` (metapath type constraint, Eq. 7 of
    DaaS_KGE-1.pdf: transition probability is 1/|neighbors of that type|).
    """
    neighbors = list(kg.successors(node)) + list(kg.predecessors(node))
    if expected_kind is not None:
        neighbors = [n for n in neighbors if _kind(kg, n) == expected_kind]
    if not neighbors:
        return None
    return rng.choice(neighbors)


def _walk_for_metapath(kg: nx.MultiDiGraph, start: str, metapath: str, length: int,
                        rng: random.Random) -> list[str]:
    """Generate one guided random walk following the type pattern in
    `metapath` (e.g. "S+" repeats station-station hops, "DSD" alternates
    drone -> station -> drone).
    """
    walk = [start]
    pattern = metapath.replace("+", "")
    pattern = pattern if pattern else _kind(kg, start)
    idx = 0
    node = start
    for _ in range(length - 1):
        expected = pattern[(idx + 1) % len(pattern)] if len(pattern) > 1 else None
        nxt = _next_step(kg, node, expected, rng)
        if nxt is None:
            nxt = _next_step(kg, node, None, rng)
        if nxt is None:
            break
        walk.append(nxt)
        node = nxt
        idx += 1
    return walk


def generate_walks(kg: nx.MultiDiGraph, metapaths: list[str] | None = None,
                    num_walks: int = 5, walk_length: int = 4, seed: int = 42) -> list[list[str]]:
    """Generate the guided random-walk corpus used to train Skip-Gram."""
    metapaths = metapaths or DEFAULT_METAPATHS
    rng = random.Random(seed)
    walks: list[list[str]] = []
    nodes = list(kg.nodes())
    for node in nodes:
        for _ in range(num_walks):
            for mp in metapaths:
                start_kind = mp[0] if mp[0] != "+" else "S"
                if _kind(kg, node) != start_kind and start_kind in KIND_OF.values():
                    continue
                walk = _walk_for_metapath(kg, node, mp, walk_length, rng)
                if len(walk) > 1:
                    walks.append(walk)
    return walks


def train_embeddings(walks: list[list[str]], dimension: int = 64, window: int = 3,
                      epochs: int = 5, seed: int = 42) -> Word2Vec:
    """Train Skip-Gram (metapath2vec's representation-learning backbone)."""
    model = Word2Vec(
        sentences=walks,
        vector_size=dimension,
        window=window,
        min_count=1,
        sg=1,              # skip-gram (as opposed to CBOW)
        negative=5,         # negative sampling
        workers=1,
        epochs=epochs,
        seed=seed,
    )
    return model


def cosine_proximity(model: Word2Vec, a: str, b: str) -> float:
    """Eq. 10 of DaaS_KGE-1.pdf: cosine similarity between two embeddings."""
    if a not in model.wv or b not in model.wv:
        return 0.0
    return float(model.wv.similarity(a, b))
