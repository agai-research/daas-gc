"""DaaS-KGE baseline: knowledge-graph-embedding-based DaaS composition.

Follows the attached "DaaS_KGE-1.pdf" (Sellami & Mezni, "Drone-as-a-Service:
Proximity-aware Composition of UAV-based Delivery Services"): builds a
multi-relational DaaS knowledge graph (DaaSKG), embeds it with
metapath2vec-style guided random walks + Skip-Gram, and composes drone
services / delivery paths from the resulting proximity subspace.
"""
