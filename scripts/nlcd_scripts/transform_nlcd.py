"""
VQ-MAR — NLCD Land-Use Suitability (Li)

Computes the per-site land-use suitability score from NLCD 2021 land
cover class and impervious surface percentage data.

Method:
  - Map NLCD class code to a base suitability score (Li_class) from a
    16-class lookup table.
  - Apply impervious modifier: Li = Li_class × (1 − imp_pct/100).
    Missing impervious_pct is treated as 0% (no penalty).
  - Set hard-exclusion flag (ei = 1) for:
      * NLCD classes 11, 23, 24, 95 (open water, developed, wetlands), or
      * Sites with impervious_pct > 30%.

Input/output paths are selected by the --grid flag:
  --grid 20 (default):
    Input:  georgia/raw/nlcd/site_nlcd_classes.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_20.csv
    Output: georgia/processed/nlcd/nlcd_scores.csv

  --grid 50:
    Input:  georgia/raw/nlcd/site_nlcd_classes_50sites.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_50sites.csv
    Output: georgia/processed/nlcd/nlcd_scores_50.csv

Usage:
    python scripts/nlcd_scripts/transform_nlcd.py              # 20-site (default)
    python scripts/nlcd_scripts/transform_nlcd.py --grid 50    # 50-site
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
RAW_SSURGO = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
RAW_NLCD = os.path.join(BASE_DIR, 'georgia', 'raw', 'nlcd')
OUT_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'nlcd')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# NLCD class-to-score mapping  (see Data Spec §5.3, Table)
# ---------------------------------------------------------------------------
SUITABILITY_SCORES = {
    11: 0.0,   # Open Water
    12: 0.0,   # Perennial Ice/Snow
    21: 0.2,   # Developed, Open Space
    22: 0.0,   # Developed, Low Intensity
    23: 0.0,   # Developed, Medium Intensity
    24: 0.0,   # Developed, High Intensity
    31: 0.8,   # Barren Land
    41: 0.6,   # Deciduous Forest
    42: 0.6,   # Evergreen Forest
    43: 0.6,   # Mixed Forest
    52: 0.8,   # Shrub/Scrub
    71: 0.9,   # Grassland/Herbaceous
    81: 0.7,   # Pasture/Hay
    82: 0.5,   # Cultivated Crops
    90: 0.1,   # Woody Wetlands
    95: 0.1,   # Emergent Herbaceous Wetlands
}

# Hard exclusion classes  (water, developed, wetlands; see Data Spec §5.4)
HARD_EXCLUDE_CLASSES = {11, 23, 24, 95}

# Impervious surface exclusion threshold
IMPERVIOUS_EXCLUSION_THRESHOLD = 30.0  # percent


def get_paths(grid_size):
    """Return (input_path, grid_path, output_path) for the given grid size."""
    if grid_size == 20:
        return (
            os.path.join(RAW_NLCD, 'site_nlcd_classes.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_20.csv'),
            os.path.join(OUT_DIR, 'nlcd_scores.csv'),
        )
    elif grid_size == 50:
        return (
            os.path.join(RAW_NLCD, 'site_nlcd_classes_50sites.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_50sites.csv'),
            os.path.join(OUT_DIR, 'nlcd_scores_50.csv'),
        )
    else:
        raise ValueError(f'Unsupported grid size: {grid_size}. Use 20 or 50.')


def main():
    parser = argparse.ArgumentParser(
        description='NLCD land-use suitability scoring with impervious surface modifier.'
    )
    parser.add_argument(
        '--grid',
        type=int,
        choices=[20, 50],
        default=20,
        help='Grid size to process (default: 20).',
    )
    args = parser.parse_args()

    nlcd_path, grid_path, out_path = get_paths(args.grid)

    print('=' * 60)
    print(f'NLCD Land-Use Suitability — grid={args.grid}')
    print('  Class-to-score mapping. Hard-exclude 11, 23, 24, 95.')
    print(f'  Impervious modifier ACTIVE '
          f'(exclusion threshold: {IMPERVIOUS_EXCLUSION_THRESHOLD}%)')
    print('=' * 60)
    print()

    # Step 1: Load raw NLCD data
    if not os.path.exists(nlcd_path):
        print(f'ERROR: {nlcd_path} not found.')
        sys.exit(1)

    nlcd = pd.read_csv(nlcd_path, dtype={'site_id': str})
    print(f'  Loaded: {len(nlcd)} sites from {os.path.basename(nlcd_path)}')

    # Step 2: Filter to the requested grid via coordinate merge.
    #   IMPORTANT: site_ids overlap across the 20- and 50-site grids (each starts
    #   from GA_001), so we must match on (latitude, longitude) to get the
    #   correct locations. This is the same fix used in
    #   build_50sites_from_existing.py.
    if os.path.exists(grid_path):
        grid = pd.read_csv(grid_path, dtype={'site_id': str})
        for col in ['latitude', 'longitude']:
            grid[col] = grid[col].round(6)
            nlcd[col] = nlcd[col].round(6)
        nlcd = grid[['site_id', 'latitude', 'longitude']].merge(
            nlcd.drop(columns=['site_id']),  # drop NLCD's site_id (may be from wrong grid)
            on=['latitude', 'longitude'],
            how='left',
        )
        print(f'  Filtered to {args.grid}-site grid: {len(nlcd)} sites')
    else:
        print(f'  WARNING: {grid_path} not found — using all sites')

    # Step 3: Map NLCD class to base suitability score (Li_class)
    nlcd['nlcd_class'] = pd.to_numeric(nlcd['nlcd_class'], errors='coerce')
    nlcd['Li_class'] = nlcd['nlcd_class'].map(SUITABILITY_SCORES)
    nlcd['Li_class'] = nlcd['Li_class'].fillna(0.5)  # Unknown classes → conservative 0.5

    # Step 4: Apply impervious modifier.
    #   Formula: Li = Li_class × (1 − imp_pct/100)
    #   Missing impervious_pct is treated as 0% (no penalty).
    #   imp_pct is clipped to [0, 100] to guard against bad data.
    if 'impervious_pct' in nlcd.columns:
        imp = pd.to_numeric(nlcd['impervious_pct'], errors='coerce').fillna(0.0)
        imp = imp.clip(lower=0.0, upper=100.0)
        nlcd['impervious_pct'] = imp
        nlcd['Li'] = nlcd['Li_class'] * (1.0 - imp / 100.0)
    else:
        print('  WARNING: impervious_pct column not found in raw input '
              '— impervious modifier skipped, Li = Li_class')
        nlcd['impervious_pct'] = 0.0
        nlcd['Li'] = nlcd['Li_class']

    # Step 5: Hard exclusion flags.
    #   ei_nlcd = 1 if either:
    #     (a) class is in HARD_EXCLUDE_CLASSES, OR
    #     (b) impervious_pct > IMPERVIOUS_EXCLUSION_THRESHOLD
    class_excluded = nlcd['nlcd_class'].apply(
        lambda x: 1 if x in HARD_EXCLUDE_CLASSES else 0
    )
    imperv_excluded = (nlcd['impervious_pct'] > IMPERVIOUS_EXCLUSION_THRESHOLD).astype(int)
    nlcd['ei_nlcd'] = ((class_excluded == 1) | (imperv_excluded == 1)).astype(int)

    # Step 6: Build output
    #   Required by unified_pipeline.py: site_id, Li, ei_nlcd, nlcd_class
    #   Plus auxiliary: latitude, longitude, nlcd_class_name (if present),
    #                   Li_class, impervious_pct (for traceability)
    out_cols = [
        'site_id', 'latitude', 'longitude',
        'nlcd_class', 'nlcd_class_name',
        'impervious_pct', 'Li_class', 'Li', 'ei_nlcd',
    ]
    if 'nlcd_class_name' not in nlcd.columns:
        out_cols.remove('nlcd_class_name')

    result = nlcd[out_cols].copy()

    # Step 7: Verify Li is in [0, 1] (catches bugs before unified pipeline runs)
    if result['Li'].min() < 0.0 or result['Li'].max() > 1.0:
        raise ValueError(
            f'Li out of [0, 1] range: '
            f'[{result["Li"].min():.4f}, {result["Li"].max():.4f}]. '
            f'Check impervious_pct values.'
        )

    # Step 8: Report
    print()
    print('Results:')
    print(f'  Sites scored: {len(result)}')
    print(f'  Li_class range:      [{result["Li_class"].min():.2f}, '
          f'{result["Li_class"].max():.2f}]')
    print(f'  Li (modified) range: [{result["Li"].min():.2f}, '
          f'{result["Li"].max():.2f}]')
    print(f'  Impervious_pct: '
          f'min={result["impervious_pct"].min():.1f}%, '
          f'max={result["impervious_pct"].max():.1f}%, '
          f'mean={result["impervious_pct"].mean():.1f}%')
    print(f'  Hard-excluded sites (ei_nlcd=1): {result["ei_nlcd"].sum()}')
    print(f'    by class:      {class_excluded.sum()}')
    print(f'    by impervious: {imperv_excluded.sum()}')
    print()
    print('  NLCD class distribution:')
    for cls, group in result.groupby('nlcd_class'):
        if pd.notna(cls):
            cls = int(cls)
            score = SUITABILITY_SCORES.get(cls, 0.5)
            excl = '** EXCLUDED **' if cls in HARD_EXCLUDE_CLASSES else ''
            print(f'    {cls:3d}  base={score:.1f}  sites={len(group):3d}  {excl}')

    # Step 9: Save
    result.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')

    return result


if __name__ == '__main__':
    main()
