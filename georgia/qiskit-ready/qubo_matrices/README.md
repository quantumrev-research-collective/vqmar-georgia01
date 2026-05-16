# QUBO Matrices

This directory contains QUBO formulations and solver outputs for the
VQ-MAR site-selection problem. Each combination of cost model and grid
size has its own subdirectory.

## Directory layout

```
qubo_matrices/
├── landscape_analysis.json     Spectral analysis (both cost models)
├── flat/                       Cost model: flat Ci = 0.5 for all sites
│   ├── n20/                    20-site grid
│   │   ├── inputs/             QUBO matrix and classical baselines
│   │   └── results/            QAOA results
│   └── n50/                    50-site grid
│       ├── inputs/
│       └── results/
└── real/                       Cost model: dollar-scale Ci, normalized to [0,1]
    ├── n20/
    │   ├── inputs/
    │   └── results/
    └── n50/
        ├── inputs/
        └── results/
```

## Top-level files

- `landscape_analysis.json` — Spectral analysis of the QUBO matrix for
  each cost model. Contains eigenvalue spectra, condition number (κ),
  spectral gap, and diagonal/off-diagonal statistics. The summary block
  reports the ratio of condition numbers between the two cost models
  and its implication for optimizer behavior at deeper QAOA layers.

## Input files (per `<cost_model>/n<N>/inputs/`)

QUBO matrix and metadata:

- `meta_<N>.json` — Run metadata: budget, parameters, site IDs
- `pairwise_<N>.json` — Pairwise interaction matrix M_ij
- `qubo_<N>.json` — Assembled QUBO matrix Q in JSON form
- `Q_<N>.npy` — QUBO matrix Q as a NumPy binary file
- `Q_<N>.csv` — QUBO matrix Q as a CSV

Classical reference solutions (used as approximation-ratio baselines
by the QAOA scripts):

- `brute_force_<N>.json` — Brute-force ground-truth solution
  (present only for N ≤ 24; for larger grids the fallback
  `best_classical_<N>.json` is used instead)
- `greedy_<N>.json` — Greedy classical baseline
- `simulated_annealing_<N>.json` — Simulated annealing result
- `genetic_algorithm_<N>.json` — Genetic algorithm result

## Solver output files (per `<cost_model>/n<N>/results/`)

Naming convention:
`<task>_n<N>_<cost_model>_p<depth>[_<extra>][_<timestamp>].json`

Three categories of file appear:

- **QAOA single runs** — one bitstring + metrics per run.
  - `qaoa_native_diagonal_n<N>_<cost_model>_p<depth>.json` — noiseless
    simulator with native diagonal QUBO encoding
  - `qaoa_vqe_estimator_n<N>_<cost_model>_p<depth>.json` — VQE +
    Estimator primitive baseline
  - `qaoa_algorithms_sampler_n<N>_<cost_model>_p<depth>.json` —
    canonical IBM Qiskit QAOA pathway (qiskit-algorithms QAOA +
    StatevectorSampler)
  - `qaoa_hardware_n<N>_<cost_model>_p<depth>_<timestamp>.json` — real
    quantum hardware run

- **QAOA top-10 variants** — for each run, the 10 lowest-energy
  bitstrings found, in the canonical schema consumed by
  `extract_metrics.py`.
  - `top10_baseline_n<N>_<cost_model>_p<depth>.json` — reference
    top-10 view from the baseline (native-diagonal) QAOA run, used as
    the comparison point for variant runs
  - `top10_cvar_n<N>_<cost_model>_p<depth>_<timestamp>.json` — CVaR
    aggregator
  - `top10_spatial_diversity_n<N>_<cost_model>_p<depth>_<timestamp>.json`
    — spatial-diversity penalty
  - `top10_hydraulic_diversity_n<N>_<cost_model>_p<depth>_<timestamp>.json`
    — hydraulic-diversity penalty
  - `top10_mps_n<N>_<cost_model>_bd<bond_dim>_p<depth>_<timestamp>.json`
    — MPS-based classical simulation (bond dimension in filename)
  - `top10_noisy_aer_n<N>_<cost_model>_p<depth>_<timestamp>.json` —
    Aer noisy simulator

- **Cross-solver report**
  - `sanity_check_n<N>_<cost_model>.json` — bounds validation and
    energy-ordering checks across all solvers run on this grid

Not every category is present in every `<cost_model>/n<N>/` directory —
slow runs (e.g., `qaoa_algorithms_sampler`, `qaoa_hardware`) are
included only for selected configurations.

## Reproducing these outputs

The input files (QUBO assembly + classical baselines) are produced by:

    python scripts/unified_pipeline.py --cost_model flat --grid 20
    python scripts/unified_pipeline.py --cost_model real --grid 50

QAOA result files are produced by the `scripts/run_qaoa_*.py` scripts,
each corresponding to one variant. Top-10 baseline reference views are
produced from `qaoa_native_diagonal_*` outputs by
`scripts/convert_results_to_schema.py`. Landscape analysis is
produced by `scripts/analyze_qubo_landscape.py`. For an aggregated
metrics CSV across all runs in this directory:

    python scripts/extract_metrics.py
