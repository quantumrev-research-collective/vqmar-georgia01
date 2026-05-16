# Processed OSM scores (intentionally empty)

This folder ships empty by design because its inputs (in
`georgia/raw/osm/`) are not redistributed — see that folder's `README.md`
for the licensing and freshness rationale. Once the raw OSM folder is
populated, the score files in this folder can be regenerated with the
transform script.

## How to populate this folder

First populate the raw OSM folder:

```bash
python scripts/osm_scripts/fetch_osm.py
```

Then run the transform script for each grid size:

```bash
python scripts/osm_scripts/transform_osm.py              # 20-site grid
python scripts/osm_scripts/transform_osm.py --grid 50    # 50-site grid
```

After both runs, this folder will contain:

- `osm_scores.csv` — per-site OSM access score `Ai` for the 20-site grid
- `osm_scores_50.csv` — same, for the 50-site grid

These are read by `scripts/unified_pipeline.py` as the OSM contribution
to the composite benefit vector `wi` and to the real-cost coefficient
`Ci` (Eq. 15).

## Further reading

For the `Ai` scoring formula (the road-access and water-source
sub-scores) and the column definitions of the output files, see
`docs/data_docs/VQ-MAR_OSM_Data_Documentation-v3.1.pdf`.
