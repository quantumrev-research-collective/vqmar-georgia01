# Raw OSM data (intentionally empty)

This folder ships empty by design. OpenStreetMap data is licensed under the
Open Database License (ODbL) with attribution and share-alike requirements,
and the underlying map changes continuously as contributors update it.
Rather than redistribute a stale snapshot, this repository ships the fetch
script and asks each user to pull fresh data themselves.

## How to populate this folder

Run the fetch script from the repository root:

```bash
python scripts/osm_scripts/fetch_osm.py
```

The script queries the public Overpass API for roads, waterways, and water
infrastructure within a 10 km radius of each candidate site. No
registration or API key is required; expected runtime is approximately
5–10 minutes.

After it completes, this folder will contain:

- `site_osm_distances.csv` and `site_osm_distances_50sites.csv` —
  per-site distances to the nearest road, waterway, and water-infrastructure
  feature for the 20-site and 50-site candidate grids.
- `osm_roads.csv`, `osm_waterways.csv`, `osm_water_infrastructure.csv` —
  the underlying feature pulls used to compute those distances.

These outputs are the inputs to `scripts/osm_scripts/transform_osm.py`,
which produces the per-site OSM access score `Ai` written to
`georgia/processed/osm/`.

## Attribution

Data fetched by the script is © OpenStreetMap contributors and is made
available under the Open Database License. Any redistribution must
preserve attribution. See
[https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).

## Further reading

For column definitions, the `Ai` scoring methodology, and the role of OSM
data in the QUBO formulation, see
`docs/data_docs/VQ-MAR_OSM_Data_Documentation-v3.1.pdf`.

