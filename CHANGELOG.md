# Changelog — revision for the reviewers' reproducibility comments

## Implementation
- `daasgc/clustering.py`: SCAN semantics (clusters grown from ε-reachable cores; strongly-connected
  cores were previously left in separate clusters, e.g. 94 clusters for a requested k = 5).
  N_ε(u) includes u, as in Eq. strongconnection. Similarities cached per network.
  Calibration of (ε, μ) runs once per network (pre-processing). New explicit k = 1 partition.
  Gateway stations added for routing; the paper's frontier set is still reported.
- `daasgc/path_search.py`: A* heuristic = straight-line distance (admissible, consistent, O(1));
  Algorithm 2 with exit-gateway enumeration, cluster sequence on the cluster adjacency graph,
  fallbacks counted and reported; pre-processed `Partition` object.
- `daasgc/hypergraph.py`: DaaS-HGC routes on the hyper-graph with virtual intra-cluster edges,
  then expands only the chosen route with local A*.
- `daasgc/delivery_model.py` (new): energy under load, pad compatibility, voltage-dependent
  charging time, pad queueing, recharging fees in cost, energy metrics, Monte-Carlo failure rate
  (replaces the synthetic failure proxy and the fixed 2.5 Wh/km energy constant).
- `daasgc/composition.py`: feasibility-aware Algorithm 3 (max feasible load per drone on the path);
  explicit sign convention of Eq. daas-score.
- `baselines/daas_kge/`: pre-processing/query split; deterministic hashing; shared feasibility model.
- `baselines/daas_sg.py`: same edge cost, heuristic and composition as DaaS-GC.

## Data
- `data/build_dataset.py`: segment counts match the paper's table (245 … 8 970), ≈ 4 pads per
  station, Euclidean segment distances; one seed per artefact (the README/`main_exp.py` seed
  collision on s100_d20 is gone). All 18 instances committed.

## Experiments and figures
- `experiments/exp1…exp4`: `--runs` (default 10) and `--seed`; protocol of README §2; exp2 and exp4
  include k = 1 and DaaS-SG / DaaS-KGE references (reviewer comment 2); timing excludes
  pre-processing for every method.
- `plots/make_figures.py` (new): the six article figures, under the article's file names.
- `results/*.json` and `figures/*.png` committed.

## Packaging
- `main_exp.py --runs N --seed S [--only exp1 …]`; scripts runnable directly (no PYTHONPATH);
  pinned `requirements.txt` (scikit-learn and pandas are no longer used); matplotlib ≥ 3.11 crash fixed;
  `daasgc.py` CLI; `tests/test_k1_consistency.py`; updated smoke test.

## Second pass — alignment with the paper's own description (Figs. 5–10)
- `daasgc/clustering.py`: functional criterion of Def. Cluster — a strong connection requires
  σ ≥ ε **and** the same pad type, so every region is functionally homogeneous (the property
  Sections 6.3.1–7.3.3 use to explain DaaS-GC's results). `FUNCTIONAL_GATE = False` restores purely
  structural clustering. Test added: every cluster has a single pad type.
- `daasgc/path_search.py`: DaaS-GC now evaluates the local path between **every entry and every
  exit** gateway of each traversed cluster (Section 6.3.4, "combinatorial situation") and keeps the
  cheapest chain (dynamic programming); region sub-graphs are materialised during pre-processing.
- `daasgc/delivery_model.py`: delivery cost includes the drone service prices (Section 6.3.1);
  battery usage = battery depletion between two full charges (Section 6.3.3, "nearly
  three-quarters of the battery capacity"), total energy drawn kept as `battery_energy_drawn_pct`.
- `plots/make_figures.py`: Fig. 7 stacks battery usage on energy efficiency as in the submitted
  figure; Figs. 9–10 use a data-driven y-range and wider boxes, as in the submitted figures.
- `daasgc/path_search.py`: `GC_ALL_ENTRIES` (default True, Section 6.3.4 reading). False runs the
  Algorithm 2 pseudo-code reading (best exit from the single current entry, cluster by cluster).
- No parameter, seed or metric was tuned toward the submitted figures.

## Figure texture
- `plots/make_figures.py`: colours, box outlines (black at 60 % opacity), median colour (#ff7f0e),
  box widths and spacing, canvas size and export resolution measured on the six submitted figures
  (Figs. 8 and 9 at 144 dpi with opaque fills, as submitted). Data unchanged.

## Figs. 6 and 9 without horizontal reference lines
- DaaS-SG does not partition the network, so its search is the k = 1 case: it is drawn in the k = 1
  group (own bar and distance marker in Fig. 6, own box in Fig. 9), labelled "1 (= DaaS-SG)", instead
  of as horizontal lines across k. DaaS-KGE is no longer drawn in Figs. 6 and 9 (its times are in Fig. 10).

## Figure texture (reference: hatched energy figure)
- `plots/make_figures.py`: one texture per method in every figure — Okabe-Ito colour-blind-safe colour
  + hatch (DaaS-GC #0072B2 `///`, DaaS-HGC #E69F00 `...`, DaaS-KGE #009E73 `xxx`, DaaS-SG #CC79A7
  `\\\\\\`), light tint with the same hatch for a second series (battery usage in Fig. 7),
  black outlines, Inter font, white outlined medians, 300 dpi. Figures remain readable in greyscale.
