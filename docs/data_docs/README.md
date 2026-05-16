# Data Documentation

This folder contains reference documentation for the data that feeds the
VQ-MAR pipeline. The content documents are PDFs; this README is an index
of what's in the folder and how the documents relate.

## Per-source documentation

Each data source contributes one sub-score to the site-benefit
coefficient `wᵢ`. The PDFs cover provenance, fetch procedure, column
definitions, quality notes, and the role the data plays in computing
its sub-score.

| Data source         | Sub-score | Weight | Document                                       |
| ------------------- | :-------: | :----: | ---------------------------------------------- |
| SSURGO (USDA NRCS)  |    Sᵢ     |  30 %  | `VQ-MAR_SSURGO_Data_Documentation-v2_1.pdf`    |
| NWIS (USGS)         |    Nᵢ     |  25 %  | `VQ-MAR_NWIS_Data_Documentation-v2_1.pdf`      |
| NLCD (USGS)         |    Lᵢ     |  15 %  | `VQ-MAR_NLCD_Data_Documentation-v2_1.pdf`      |
| NOAA (NCEI)         |    Cᵢ     |  15 %  | `VQ-MAR_NOAA_Data_Documentation-v3_2.pdf`      |
| OSM (OpenStreetMap) |    Aᵢ     |  15 %  | `VQ-MAR_OSM_Data_Documentation-v3_1.pdf`       |

## Project-level documents

Two PDFs cover the pipeline as a whole rather than one source at a time:

- **`VQ-MAR_Georgia_Data_Specification-v2_1.pdf`** — *what* data is
  required. Defines the QUBO Hamiltonian, the `wᵢ` benefit formula, the
  risk penalty, the cost-coefficient models, and the Qiskit mapping.
- **`VQ-MAR_Georgia_Data_Access_Guide-v4_3.pdf`** — *how* to fetch each
  data source end-to-end: prerequisites, scripts, expected runtimes,
  verification checklist.

## How the documents relate

The Georgia Data Specification is the canonical source for the QUBO
formulation and any equations referenced in the per-source PDFs. The
Data Access Guide is the operational companion to the Specification.

For the study-area definition (Dougherty Plain, Upper Floridan Aquifer,
candidate-grid construction), see the top-level repository `README.md`
or §1 of the Data Specification.
