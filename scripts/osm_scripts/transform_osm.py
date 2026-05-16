"""
VQ-MAR — OSM Infrastructure Access (Ai)

Computes the per-site infrastructure access score from OpenStreetMap
distance data (major roads, waterways, and water infrastructure).

Method:
  - Road access sub-score:
      road_access_score = 1.0 - clip(dist_road_m, 0, 10_000) / 10_000
  - Water source sub-score (preferred): read directly from the raw input
    column water_source_score, which is precomputed by the
    build_50sites_from_existing.py upstream step. If that column is
    absent, fall back to computing it on the fly from dist_waterway_m
    using the same clip-and-scale formula.
  - Composite: Ai = 0.70 * road_access_score + 0.30 * water_source_score
  - Sites with NaN dist_road_m (Overpass API tile timeouts) are
    defended by filling to CLIP_DISTANCE_M = 10_000 m before scoring,
    which yields a road_access_score of 0.0 for those sites and is
    logged with a warning.

Architectural note: the real cost coefficient Ci (Eq. 15 of the paper)
is computed in unified_pipeline.py, not in this script. Ci requires
data from SSURGO (wtdepannmin) and NLCD (Li) that the OSM transform
does not load; keeping Ci out of this script preserves a clean
separation between sub-score producers (the transform layer) and
composite cost producers (the pipeline layer).

Input/output paths are selected by the --grid flag:
  --grid 20 (default):
    Input:  georgia/raw/osm/site_osm_distances.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_20.csv
    Output: georgia/processed/osm/osm_scores.csv

  --grid 50:
    Input:  georgia/raw/osm/site_osm_distances_50sites.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_50sites.csv
    Output: georgia/processed/osm/osm_scores_50.csv

Usage:
    python scripts/osm_scripts/transform_osm.py              # 20-site (default)
    python scripts/osm_scripts/transform_osm.py --grid 50    # 50-site
"""

import argparse
import pandas as pd
import os
import sys

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# --- Path resolution (genericized for public repo) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)
RAW_OSM = os.path.join(BASE_DIR, 'georgia', 'raw', 'osm')
RAW_SSURGO = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
OUT_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'osm')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants  (see Data Spec §7.4)
# ---------------------------------------------------------------------------
CLIP_DISTANCE_M = 10_000  # 10 km normalization ceiling

# Composite weights for Ai
W_ROAD_ACCESS = 0.70
W_WATER_SOURCE = 0.30


def get_paths(grid_size):
    """Return (input_path, grid_path, output_path) for the given grid size."""
    if grid_size == 20:
        return (
            os.path.join(RAW_OSM, 'site_osm_distances.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_20.csv'),
            os.path.join(OUT_DIR, 'osm_scores.csv'),
        )
    elif grid_size == 50:
        return (
            os.path.join(RAW_OSM, 'site_osm_distances_50sites.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_50sites.csv'),
            os.path.join(OUT_DIR, 'osm_scores_50.csv'),
        )
    else:
        raise ValueError(f'Unsupported grid size: {grid_size}. Use 20 or 50.')


def main():
    parser = argparse.ArgumentParser(
        description='OSM infrastructure access scoring (road + water-source components).'
    )
    parser.add_argument(
        '--grid',
        type=int,
        choices=[20, 50],
        default=20,
        help='Grid size to process (20 or 50; default 20).',
    )
    args = parser.parse_args()

    osm_path, grid_path, out_path = get_paths(args.grid)

    print('=' * 60)
    print(f'OSM Infrastructure Access — grid={args.grid}')
    print(f'  Composite Ai = '
          f'{W_ROAD_ACCESS}*road_access + {W_WATER_SOURCE}*water_source')
    print('=' * 60)
    print()

    # Step 1: Load raw OSM distances
    if not os.path.exists(osm_path):
        print(f'ERROR: {osm_path} not found.')
        sys.exit(1)
    osm = pd.read_csv(osm_path, dtype={'site_id': str})
    print(f'  Loaded: {len(osm)} sites from {os.path.basename(osm_path)}')

    # Step 2: Filter to the requested grid via coordinate merge.
    #   IMPORTANT: site_ids may overlap across grid sizes, so we must
    #   match on (latitude, longitude) to get the correct locations.
    if os.path.exists(grid_path):
        grid = pd.read_csv(grid_path, dtype={'site_id': str})
        for col in ['latitude', 'longitude']:
            grid[col] = grid[col].round(6)
            osm[col] = osm[col].round(6)
        osm = grid[['site_id', 'latitude', 'longitude']].merge(
            osm.drop(columns=['site_id']),
            on=['latitude', 'longitude'],
            how='left',
        )
        print(f'  Filtered to {args.grid}-site grid: {len(osm)} sites')
    else:
        print(f'  WARNING: {grid_path} not found — using all sites')

    # Step 3: Defend against missing road distances.
    #   For --grid 50 this should never happen: build_50sites_from_existing.py
    #   fills missing distances before writing the 50-site input file.
    #   For --grid 20 it may happen, since the 20-site input file is the raw
    #   Overpass output and a tile timeout (HTTP 504) leaves NaN distances.
    osm['dist_road_m'] = pd.to_numeric(osm['dist_road_m'], errors='coerce')
    n_missing_road = osm['dist_road_m'].isna().sum()
    if n_missing_road > 0:
        print(f'  WARNING: {n_missing_road} sites missing dist_road_m. '
              f'Defaulting to {CLIP_DISTANCE_M} m (worst case).')
        osm['dist_road_m'] = osm['dist_road_m'].fillna(CLIP_DISTANCE_M)

    # Step 4: Compute road access score
    osm['road_access_score'] = 1.0 - (
        osm['dist_road_m'].clip(0, CLIP_DISTANCE_M) / CLIP_DISTANCE_M
    )

    # Step 5: Determine water_source_score.
    #   If the raw input already has a precomputed water_source_score column
    #   (the build_50sites_from_existing.py path), use it directly. Otherwise
    #   compute it on the fly from dist_waterway_m using the same
    #   clip-and-scale formula as road_access_score.
    if 'water_source_score' in osm.columns:
        osm['water_source_score'] = pd.to_numeric(
            osm['water_source_score'], errors='coerce'
        ).fillna(0.0).clip(0.0, 1.0)
        water_source_origin = 'read from raw input'
    elif 'dist_waterway_m' in osm.columns:
        dist_w = pd.to_numeric(
            osm['dist_waterway_m'], errors='coerce'
        ).fillna(CLIP_DISTANCE_M)
        osm['water_source_score'] = 1.0 - (
            dist_w.clip(0, CLIP_DISTANCE_M) / CLIP_DISTANCE_M
        )
        water_source_origin = 'computed from dist_waterway_m'
    else:
        print('  WARNING: no water_source_score or dist_waterway_m in raw input. '
              'Defaulting all sites to 0.0 (worst).')
        osm['water_source_score'] = 0.0
        water_source_origin = 'defaulted to 0.0 (no source data)'
    print(f'  water_source_score: {water_source_origin}')

    # Step 6: Composite Ai
    osm['Ai'] = (
        W_ROAD_ACCESS * osm['road_access_score']
        + W_WATER_SOURCE * osm['water_source_score']
    )

    # Step 7: Build output
    #   Required by unified_pipeline.py: site_id, Ai, dist_road_m
    #   Plus auxiliary: latitude, longitude, sub-scores, raw distances
    out_cols = ['site_id', 'latitude', 'longitude', 'dist_road_m']
    for col in ['dist_waterway_m', 'dist_water_infra_m']:
        if col in osm.columns:
            out_cols.append(col)
    out_cols += ['road_access_score', 'water_source_score', 'Ai']

    result = osm[out_cols].copy()

    # Step 8: Verify Ai in [0, 1]
    if result['Ai'].min() < 0.0 or result['Ai'].max() > 1.0:
        raise ValueError(
            f'Ai out of [0, 1]: '
            f'[{result["Ai"].min():.4f}, {result["Ai"].max():.4f}]'
        )

    # Step 9: Report
    print()
    print('Results:')
    print(f'  Sites scored: {len(result)}')
    print(f'  road_access_score:  '
          f'mean={result["road_access_score"].mean():.3f}, '
          f'range=[{result["road_access_score"].min():.3f}, '
          f'{result["road_access_score"].max():.3f}]')
    print(f'  water_source_score: '
          f'mean={result["water_source_score"].mean():.3f}, '
          f'range=[{result["water_source_score"].min():.3f}, '
          f'{result["water_source_score"].max():.3f}]')
    print(f'  Ai (composite):     '
          f'mean={result["Ai"].mean():.3f}, '
          f'range=[{result["Ai"].min():.3f}, '
          f'{result["Ai"].max():.3f}]')
    dist = result['dist_road_m'].dropna()
    print(f'  Road distance: mean={dist.mean():.0f} m, '
          f'median={dist.median():.0f} m')
    print(f'  Within 1 km of road: {(dist < 1000).sum()}, '
          f'within 5 km: {(dist < 5000).sum()}')

    # Step 10: Save
    result.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')

    return result


if __name__ == '__main__':
    main()
