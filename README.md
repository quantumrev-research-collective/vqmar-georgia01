# VQ-MAR: Georgia Aquifer Recharge Pilot

Reference implementation for **VQ-MAR**, a variational quantum-classical hybrid framework for Managed Aquifer Recharge (MAR) site selection. This repository contains the data acquisition pipeline, scoring transforms, QUBO assembly, classical solvers, and QAOA pathway used in the Georgia pilot study.

The pipeline is applied to the Dougherty Plain in southwest Georgia, covering the Upper Floridan Aquifer system across ten counties. The QAOA pathway is executed on Qiskit Aer simulators and validated on the IBM Kingston superconducting QPU at `N = 20` and `N = 50`.

---

## Citation

This repository accompanies the paper:

> Aseeri, S. A., Burgett, J., Melmer Stiefkens, J., Taylor, T., & Milewski, A. M. (2026). *VQ-MAR: A Variational Quantum-Classical Hybrid Framework for Managed Aquifer Recharge Site Selection.* 2026 IEEE International Conference on Quantum Computing and Engineering (QCE), Quantum End-to-End Hybrid Case Studies (QECS) Track.

---

## Study Area

- **Region:** Dougherty Plain, southwest Georgia
- **Bounding box:** 31.60°N–32.50°N, 84.60°W–83.60°W (~100 km × 95 km)
- **Aquifer:** Upper Floridan Aquifer system
- **Counties:** Sumter, Lee, Terrell, Webster, Schley, Macon, Dooly, Crisp, Worth, Dougherty
- **Candidate site grids:** Regular lattices at `N = 20` and `N = 50` for the QUBO scaling study

---

## Data Sources

Five publicly funded, open-access data sources feed the VQ-MAR QUBO Hamiltonian. Each contributes one of the linear sub-scores in the site benefit coefficient:

| Source                  | Sub-score | Weight | Role                         | API / Service                                 |
| ----------------------- | --------- | ------ | ---------------------------- | --------------------------------------------- |
| **SSURGO** (USDA NRCS)  | Sᵢ        | 30%    | Soil infiltration capacity   | Soil Data Access REST endpoint                |
| **NWIS** (USGS)         | Nᵢ        | 25%    | Groundwater storage capacity | Water Services REST API via `dataretrieval`   |
| **NLCD** (USGS)         | Lᵢ        | 15%    | Land-use suitability         | MRLC GeoServer WMS                            |
| **NOAA NCEI**           | Cᵢ        | 15%    | Climate and recharge timing  | NCEI Access Data Service v1 (tokenless)       |
| **OSM** (OpenStreetMap) | Aᵢ        | 15%    | Infrastructure access        | Overpass API                                  |

Total raw data is approximately 40 MB. Total cost: $0. Total active collection time: ~20 minutes.

---

## Repository Structure

```
vqmar-georgia01/
├── .gitignore
├── README.md
├── LICENSE                                          # Apache License 2.0
├── NOTICE                                           # Copyright and attribution
├── requirements.txt
├── docs/
│   └── data_docs/
│       ├── README.md
│       ├── VQ-MAR_Georgia_Data_Access_Guide-v4_3.pdf
│       ├── VQ-MAR_Georgia_Data_Specification-v2_1.pdf
│       ├── VQ-MAR_SSURGO_Data_Documentation-v2_1.pdf
│       ├── VQ-MAR_NWIS_Data_Documentation-v2_1.pdf
│       ├── VQ-MAR_NLCD_Data_Documentation-v2_1.pdf
│       ├── VQ-MAR_NOAA_Data_Documentation-v3_2.pdf
│       └── VQ-MAR_OSM_Data_Documentation-v3_1.pdf
├── georgia/
│   ├── raw/                                         # Federal-source raw data
│   │   ├── ssurgo/   (soil properties, texture, water table, mukey grids)
│   │   ├── nwis/     (well metadata, daily groundwater levels)
│   │   ├── nlcd/     (land cover classes, impervious surface)
│   │   ├── noaa/     (Albany station daily climate)
│   │   └── osm/      (regenerated at runtime; ODbL — see folder README)
│   ├── processed/                                   # Per-source scoring outputs
│   │   ├── ssurgo/   (ssurgo_scores.csv, ssurgo_scores_50.csv)
│   │   ├── nwis/     (nwis_scores.csv,   nwis_scores_50.csv)
│   │   ├── nlcd/     (nlcd_scores.csv,   nlcd_scores_50.csv)
│   │   ├── noaa/     (noaa_scores.csv,   noaa_scores_50.csv)
│   │   ├── osm/      (regenerated at runtime; see folder README)
│   │   ├── output_files_reference.pdf               # Schema reference for the files below
│   │   ├── sites_{20,50}_scored_{flat,real}.csv     # Top-level consolidated scores
│   │   └── results_{20,50}_{flat,real}.csv          # Solver comparison tables
│   └── qiskit-ready/                                # QUBO matrices and analysis outputs
│       ├── qubo_matrices/
│       │   ├── flat/                                # Uniform-cost QUBO + solver outputs
│       │   └── real/                                # Heterogeneous-cost QUBO + solver outputs
│       ├── site_metadata/
│       │   ├── flat/                                # Site metadata, flat cost model
│       │   └── real/                                # Site metadata, real cost model
│       ├── portfolio_analysis/                      # N=20 portfolio characterization outputs
│       ├── portfolio_analysis_n20_mps_baseline/     # N=20 MPS χ=64 baseline run
│       └── portfolio_analysis_n50/                  # N=50 variant runs (CVaR, spatial, hydraulic, MPS, noisy)
└── scripts/
    ├── unified_pipeline.py                          # QUBO assembly + classical solvers
    ├── build_50sites_from_existing.py               # Build 50-site dataset from existing files
    ├── integration/
    │   └── build_sites_scored.py                    # Per-source score consolidation
    ├── ssurgo_scripts/
    │   ├── fetch_ssurgo.py
    │   ├── fetch_ssurgo_spatial.py
    │   └── transform_ssurgo.py
    ├── nwis_scripts/
    │   ├── fetch_nwis.py
    │   └── transform_nwis.py
    ├── nlcd_scripts/
    │   ├── fetch_nlcd.py
    │   └── transform_nlcd.py
    ├── noaa_scripts/
    │   ├── fetch_noaa.py
    │   └── transform_noaa.py
    ├── osm_scripts/
    │   ├── fetch_osm.py
    │   └── transform_osm.py
    └── qaoa/
        ├── analyze_qubo_landscape.py                # Energy landscape on β/γ grid
        ├── convert_results_to_schema.py             # Solver output schema normalization
        ├── qiskit_qubo.py                           # QuadraticProgram → QUBO builder
        ├── run_qaoa_algorithms_sampler.py           # Sampler-based QAOA (qiskit-algorithms)
        ├── run_qaoa_cvar.py                         # CVaR-aggregated QAOA (α = 0.25)
        ├── run_qaoa_hardware.py                     # IBM Kingston QPU execution
        ├── run_qaoa_hydraulic_diversity.py          # Hydraulic-diversity variant
        ├── run_qaoa_mps.py                          # Matrix Product State simulator (χ = 64)
        ├── run_qaoa_native_diagonal.py              # Native-gate diagonal QAOA
        ├── run_qaoa_noisy_aer.py                    # Noisy Aer (FakeBrisbane, 8192 shots)
        ├── run_qaoa_spatial_diversity.py            # Spatial-diversity variant
        └── run_qaoa_vqe_estimator.py                # Estimator-based VQE pathway
```

---

## Prerequisites

- **OS:** Ubuntu 22.04 / 24.04 (not strictly required; was the development environment)
- **Python:** 3.11
- **System libraries:** `gdal-bin`, `libgdal-dev`, `libspatialindex-dev`

No API tokens or registrations are required; all five data sources are tokenless as of 2026.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/quantumrev-research-collective/vqmar-georgia01.git
cd vqmar-georgia01
```

Install system dependencies:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev gdal-bin libgdal-dev libspatialindex-dev -y
```

Create and activate the virtual environment:

```bash
python3.11 -m venv vqmar-env
source vqmar-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No further configuration is required. The scripts in this repository derive their data paths from each script's own location on disk, so every command in the Workflow section below works from wherever you cloned the repo — no environment variables to set.

For non-default setups — running scripts as a service, splitting code and data across volumes, CI environments — you can override the data root by exporting `VQMAR_BASE` before running any script:

```bash
export VQMAR_BASE=/path/to/data/root
```

With this set, scripts read inputs from and write outputs to `$VQMAR_BASE/georgia/...` instead of paths relative to their own location. Most users will not need this.

---

## Workflow

The pipeline has four stages: **(1) raw data acquisition**, **(2) 50-site dataset assembly**, **(3) per-source scoring**, and **(4) QUBO assembly with classical solvers**.

### Stage 1 — Raw data acquisition (~20 minutes)

Run the fetch scripts in this order. Step 1 must come first (it generates the candidate site grids that all other scripts depend on). Steps 2–6 can run in any order after that.

```bash
source vqmar-env/bin/activate

# 1. SSURGO — generate site grids and assign soil map units (~5 min)
python scripts/ssurgo_scripts/fetch_ssurgo_spatial.py

# 2. SSURGO — fetch soil properties for those map units (~2 min)
python scripts/ssurgo_scripts/fetch_ssurgo.py

# 3. NWIS — well metadata and daily groundwater level time series (~5 min)
python scripts/nwis_scripts/fetch_nwis.py

# 4. NLCD — land cover class and impervious surface at each site (~2 min)
python scripts/nlcd_scripts/fetch_nlcd.py

# 5. NOAA — daily precipitation and temperature, 2001–2025 (~1 min, single CSV request)
python scripts/noaa_scripts/fetch_noaa.py

# 6. OSM — road, waterway, and water infrastructure distances (~5–10 min)
python scripts/osm_scripts/fetch_osm.py
```

Total elapsed time: approximately 20 minutes.

### Stage 2 — Build the 50-site dataset (~10 seconds)

The 50-site grid is built by filtering the existing raw data to the 50 coordinates in `site_mukeys_50.csv`. No new API calls are made; the script applies the OSM silent-failure patch (any 0.0 or NaN distance is reassigned to 10 000 m) and produces `*_50sites.csv` variants of the raw input files.

```bash
python scripts/build_50sites_from_existing.py
```

### Stage 3 — Per-source scoring transforms

Each `transform_*.py` script reads from `georgia/raw/{source}/` and writes per-source scores to `georgia/processed/{source}/`. Each script accepts a `--grid {20,50}` flag (default 20):

```bash
# 20-site grid (default)
python scripts/ssurgo_scripts/transform_ssurgo.py
python scripts/nwis_scripts/transform_nwis.py
python scripts/nlcd_scripts/transform_nlcd.py
python scripts/noaa_scripts/transform_noaa.py
python scripts/osm_scripts/transform_osm.py

# 50-site grid
python scripts/ssurgo_scripts/transform_ssurgo.py --grid 50
python scripts/nwis_scripts/transform_nwis.py     --grid 50
python scripts/nlcd_scripts/transform_nlcd.py     --grid 50
python scripts/noaa_scripts/transform_noaa.py     --grid 50
python scripts/osm_scripts/transform_osm.py       --grid 50
```

Each transform implements the full-fidelity scoring used in the paper:

- **SSURGO** — three-component infiltration score (`0.50·ksat + 0.30·hydgrp + 0.20·texcl`) with a clogging modifier from clay and organic-matter content. NULL water-table values are interpolated from the nearest NWIS well.
- **NWIS** — storage capacity (70%) plus a linear-regression trend score over 1990–2026 depth-to-water records (30%) plus a responsiveness bonus from the standard deviation of depth observations.
- **NLCD** — land-use suitability with an impervious-surface modifier and hard exclusion of developed and open-water classes.
- **NOAA** — climate priority computed from Hargreaves-Samani PET (FAO-56 Eq. 21), the Walsh-Lawler precipitation concentration index, and heavy-rainfall day frequency. Years 2000 and 2004 are excluded for data-quality reasons.
- **OSM** — composite access score (`0.70·road + 0.30·water_source`) with the silent-failure patch already applied during Stage 2.

### Stage 4 — QUBO assembly and classical solvers (under a minute per run)

The unified pipeline reads the per-source score files, computes the composite benefit and cost coefficients, builds the QUBO matrix, and runs the classical solvers (greedy, simulated annealing, genetic algorithm, plus brute-force enumeration at small `N`). It takes `--cost_model {flat,real}` and `--grid {20,50}`:

```bash
# 20-site grid (default) — both cost models
python scripts/unified_pipeline.py --cost_model flat --grid 20
python scripts/unified_pipeline.py --cost_model real --grid 20

# 50-site grid — both cost models
python scripts/unified_pipeline.py --cost_model flat --grid 50
python scripts/unified_pipeline.py --cost_model real --grid 50
```

Brute-force enumeration is auto-skipped when `N > 24` (2²⁵+ combinations becomes impractical). At `N = 50` the pipeline falls back to the best classical heuristic as the approximation-ratio (AR) reference. The reference source is logged to console during the sanity-check phase and persisted per-row in `results_{N}_{cost_model}.csv` via three columns: `AR`, `AR Reference Source`, `AR Reference Energy`. At `N = 20` the AR reference is always `brute_force` (the true global minimum); at `N = 50` it is typically `simulated_annealing`. For a column-level reference of the consolidated and results files produced at this stage, see `georgia/processed/output_files_reference.pdf`.

Outputs are written to:

- `georgia/processed/sites_{N}_scored_{flat,real}.csv` — consolidated site scores with `wᵢ`, `Cᵢ`, `eᵢ`
- `georgia/processed/results_{N}_{flat,real}.csv` — per-solver comparison table with AR reference columns
- `georgia/qiskit-ready/qubo_matrices/{flat,real}/` — QUBO matrix (JSON, NPY, and CSV), per-solver result JSONs, pairwise interaction matrix, and sanity check JSON, all suffixed by grid size (e.g. `qubo_20.json`, `qubo_50.json`)
- `georgia/qiskit-ready/site_metadata/{flat,real}/` — site metadata copies for Qiskit ingestion

All four runs share an identical scoring layer and differ only in grid size (`N`) and per-site cost coefficient `Cᵢ`.

### Scope: numerical reproduction vs. figure generation

This repository ships the data acquisition pipeline, scoring transforms, QUBO assembly, and solver implementations needed to **reproduce the numerical results** in the paper. The outputs of the pipeline — `results_{N}_{cost_model}.csv`, `sites_{N}_scored_{cost_model}.csv`, the JSONs under `qiskit-ready/`, and the per-variant outputs under `qiskit-ready/portfolio_analysis*/` — contain every quantity reported in the paper's tables and underlying every figure.

Figure-generation scripts are deliberately **not** included. Users wishing to recreate the paper's figures can do so directly from the data files above using their preferred plotting library. This scoping decision keeps the dependency footprint minimal and the reproducibility surface narrow: a verified install plus a successful pipeline run reproduces the paper's numerical claims, with no figure-generation code path to debug.

---

## Documentation

Reference documentation for each data source, the full data specification, and the end-to-end access guide live in `docs/data_docs/`. See `docs/data_docs/README.md` for an index of the available PDFs and how they relate to one another.

---

## QUBO Formulation

The VQ-MAR Hamiltonian encodes MAR site selection as a binary optimization problem:

```
H(x) = −Σ wᵢ·xᵢ + λ(Σ Cᵢ·xᵢ − B)² + μΣ eᵢ·xᵢ + Σ Mᵢⱼ·xᵢ·xⱼ
```

The site benefit `wᵢ` combines five domain sub-scores with a multiplicative risk modifier:

```
wᵢ           = wᵢ,base · (1 − risk_penalty)
wᵢ,base      = 0.30·Sᵢ + 0.25·Nᵢ + 0.15·Lᵢ + 0.15·Cᵢ + 0.15·Aᵢ
risk_penalty = clip(0.10·impervious_risk + 0.05·maintenance_risk, 0, 0.30)
```

where:

- **Sᵢ** — SSURGO infiltration (three-component + clogging modifier)
- **Nᵢ** — NWIS storage capacity (storage + trend + responsiveness)
- **Lᵢ** — NLCD land-use suitability (with impervious modifier)
- **Cᵢ** — NOAA climate priority (Hargreaves-Samani PET sub-scores)
- **Aᵢ** — OSM infrastructure access (road + water source)

The cost coefficient `Cᵢ` is evaluated under two models. The uniform model assigns `Cᵢ = 0.5` to all sites. The heterogeneous model uses Eq. (15) of the paper:

```
Cᵢ = 0.20 + 0.30·D̃_wt + 0.30·(1 − L̃ᵢ) + 0.20·(1 − Ãᵢ)
```

where `D̃_wt`, `L̃ᵢ`, and `Ãᵢ` are normalized water-table depth, land-use suitability, and road access. Both cost models are evaluated by `unified_pipeline.py` to support the cost-sensitivity analysis discussed in the paper.

The QUBO is solved by classical baselines (greedy, simulated annealing, genetic algorithm) and a quantum-compatible QAOA pathway via Qiskit on IBM Quantum hardware. QAOA variants under `scripts/qaoa/` include CVaR aggregation, native-diagonal execution, MPS simulation at χ = 64, noisy Aer simulation, and the spatial- and hydraulic-diversity formulations analyzed in the paper.

---

## Authors

- **Samar A. Aseeri** — King Abdullah University of Science & Technology
- **Jeff Burgett** — QuantumRev
- **Julián Melmer Stiefkens** — Universidad de Buenos Aires
- **Tom Taylor** — QuantumRev
- **Adam M. Milewski** — University of Georgia

---

## Acknowledgments

Sponsored by QuantumRev. Hardware time provided by IBM Quantum.

---

## License

**Code:** Apache License 2.0 — see [`LICENSE`](LICENSE).

**Data:** Raw data in `georgia/raw/` retains the licensing of its original source. SSURGO, NWIS, NLCD, and NOAA are U.S. federal public domain. OpenStreetMap-derived data, generated by the pipeline at runtime, is subject to the Open Database License (ODbL) per OSM contributor terms (https://opendatacommons.org/licenses/odbl/).

---

## Contact

This repository accompanies a paper currently under peer review for IEEE QCE 2026. During the review period, the authors are not responding to GitHub issues or other inquiries here. Reviewers with questions about this repository should contact the authors through the IEEE QCE 2026 review or submission system.
