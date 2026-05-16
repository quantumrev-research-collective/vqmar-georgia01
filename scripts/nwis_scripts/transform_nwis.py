"""
VQ-MAR — NWIS Storage Capacity (Ni)

Computes the per-site storage capacity score from USGS NWIS groundwater
depth-to-water time series, restricted to the Upper Floridan Aquifer
(nat_aqfr_cd = 'S400FLORDN').

Method:
  - For each well, compute mean depth-to-water, std, and the linear-
    regression slope of depth vs. time (ft/yr).
  - Drop wells with fewer than MIN_OBS_COUNT observations; compute trend
    slopes only for wells spanning at least MIN_TREND_YEARS distinct
    calendar years (otherwise slope is NaN).
  - Assign each candidate site the nearest qualifying well within
    MAX_WELL_DIST_M = 15 km. Sites with no well within 15 km receive
    Ni = DEFAULT_SCORE (0.5).
  - Compose three sub-scores into the final Ni:
      storage_score:        min(mean_depth_ft / 30, 1.0)
      trend_score:          favors stable/recovering aquifers,
                            penalizes declining ones (see note below)
      responsiveness_bonus: rewards wells with high seasonal variability
                            (aquifer-surface connectivity, good for MAR);
                            saturates at +0.10
  - Ni = clip(0.70 * storage + 0.30 * trend + responsiveness_bonus, 0, 1)

INTERPRETATION NOTE — sign of the trend:
  "Depth-to-water" is measured downward from the surface. An INCREASING
  slope means the water table is FALLING (aquifer being depleted).
  This script penalizes positive slopes and rewards stable-or-recovering
  wells. The opposite interpretation (reward depleted aquifers because
  they "need" MAR) is also defensible. To flip the convention, change
  trend_score() to return `1.0 - max(0, ...)` for slope > 0. The default
  is the conservative "favor sustainable contexts" interpretation;
  results should report the convention used.

Input/output paths are selected by the --grid flag:
  --grid 20 (default):
    Sites:  georgia/raw/nwis/nwis_gw_sites.csv
    Levels: georgia/raw/nwis/nwis_gw_levels.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_20.csv
    Output: georgia/processed/nwis/nwis_scores.csv

  --grid 50:
    Sites:  georgia/raw/nwis/nwis_gw_sites_50sites.csv
    Levels: georgia/raw/nwis/nwis_gw_levels_50sites.csv
    Grid:   georgia/raw/ssurgo/candidate_grid_50sites.csv
    Output: georgia/processed/nwis/nwis_scores_50.csv

Note: NWIS queries are county-scoped (by FIPS, not per-site), so the
well-level data is invariant in site count. The 20-site and 50-site
modes differ only in the candidate grid file and the output filename;
either site count may share the same underlying well source file.

Usage:
    python scripts/nwis_scripts/transform_nwis.py              # 20-site (default)
    python scripts/nwis_scripts/transform_nwis.py --grid 50    # 50-site
"""

import argparse
import math
import numpy as np
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
RAW_NWIS = os.path.join(BASE_DIR, 'georgia', 'raw', 'nwis')
RAW_SSURGO = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
OUT_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'nwis')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — well selection and storage scoring
# ---------------------------------------------------------------------------
TARGET_DEPTH_FT = 30.0          # depth-to-water target for storage_score = 1.0
MAX_WELL_DIST_M = 15_000        # 15 km — beyond this, assign default
DEFAULT_SCORE = 0.5             # Ni when no well is within MAX_WELL_DIST_M
FLORIDAN_CODE = 'S400FLORDN'    # Upper Floridan national aquifer code
MIN_OBS_COUNT = 365             # min daily observations to trust a well

# ---------------------------------------------------------------------------
# Constants — trend and responsiveness scoring
# ---------------------------------------------------------------------------
# Composite weights for Ni
W_STORAGE = 0.70
W_TREND = 0.30

# Trend computation parameters
MIN_TREND_YEARS = 3             # require >= 3 distinct calendar years for trend
TREND_NEUTRAL_SCORE = 0.5       # used when trend cannot be computed

# Slope (ft/yr) at which trend_score reaches 0.
# +0.3 ft/yr means the water table is dropping by 0.3 ft per year,
# which is the boundary at which we consider an aquifer "rapidly declining."
TREND_DECLINE_THRESHOLD_FT_YR = 0.3

# Responsiveness bonus parameters
# Standard deviation of depth-to-water (ft) at which the bonus saturates.
# 5 ft std means roughly 10 ft of seasonal swing — strong aquifer-surface
# connectivity, which is favorable for MAR effectiveness.
RESPONSIVENESS_SATURATION_FT = 5.0
RESPONSIVENESS_BONUS_MAX = 0.10  # bonus tops out at +0.10 on Ni


def get_paths(grid_size):
    """Return (sites_path, levels_path, grid_path, output_path)."""
    if grid_size == 20:
        return (
            os.path.join(RAW_NWIS, 'nwis_gw_sites.csv'),
            os.path.join(RAW_NWIS, 'nwis_gw_levels.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_20.csv'),
            os.path.join(OUT_DIR, 'nwis_scores.csv'),
        )
    elif grid_size == 50:
        # NWIS queries are county-scoped, so the underlying well data is
        # invariant in site count. Fall back to the 20-site source file
        # when the _50sites variant is missing on disk.        
        sites_50 = os.path.join(RAW_NWIS, 'nwis_gw_sites_50sites.csv')
        sites_20 = os.path.join(RAW_NWIS, 'nwis_gw_sites.csv')
        levels_50 = os.path.join(RAW_NWIS, 'nwis_gw_levels_50sites.csv')
        levels_20 = os.path.join(RAW_NWIS, 'nwis_gw_levels.csv')
        return (
            sites_50 if os.path.exists(sites_50) else sites_20,
            levels_50 if os.path.exists(levels_50) else levels_20,
            os.path.join(RAW_SSURGO, 'candidate_grid_50sites.csv'),
            os.path.join(OUT_DIR, 'nwis_scores_50.csv'),
        )
    else:
        raise ValueError(f'Unsupported grid size: {grid_size}. Use 20 or 50.')


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between two lat/lon points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Score helpers (storage + trend + responsiveness)
# ---------------------------------------------------------------------------
def storage_score(mean_depth_ft):
    """Storage capacity sub-score: min(depth/30, 1.0)."""
    if pd.isna(mean_depth_ft) or mean_depth_ft <= 0:
        return 0.0
    return min(mean_depth_ft / TARGET_DEPTH_FT, 1.0)


def trend_score(slope_ft_yr):
    """
    Trend sub-score: penalize aquifer decline, reward stability/recovery.

      slope <= 0 (stable or recovering):  score = 1.0
      0 < slope < TREND_DECLINE_THRESHOLD: linear from 1.0 down to 0.0
      slope >= TREND_DECLINE_THRESHOLD:    score = 0.0
      slope is NaN:                        score = TREND_NEUTRAL_SCORE (0.5)
    """
    if pd.isna(slope_ft_yr):
        return TREND_NEUTRAL_SCORE
    if slope_ft_yr <= 0:
        return 1.0
    return max(0.0, 1.0 - slope_ft_yr / TREND_DECLINE_THRESHOLD_FT_YR)


def responsiveness_bonus(depth_std_ft):
    """
    Responsiveness bonus: reward wells with high seasonal variability.
    Saturates at RESPONSIVENESS_BONUS_MAX (default +0.10) when std exceeds
    RESPONSIVENESS_SATURATION_FT (default 5.0 ft).
    """
    if pd.isna(depth_std_ft) or depth_std_ft <= 0:
        return 0.0
    saturation = min(depth_std_ft / RESPONSIVENESS_SATURATION_FT, 1.0)
    return saturation * RESPONSIVENESS_BONUS_MAX


def compose_Ni(mean_depth, slope, depth_std):
    """Combine the three sub-scores into the final Ni (clipped to [0,1])."""
    s = storage_score(mean_depth)
    t = trend_score(slope)
    b = responsiveness_bonus(depth_std)
    raw = W_STORAGE * s + W_TREND * t + b
    return max(0.0, min(raw, 1.0)), s, t, b


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_sites(grid_path):
    if not os.path.exists(grid_path):
        print(f'ERROR: {grid_path} not found.')
        sys.exit(1)
    sites = pd.read_csv(grid_path, dtype={'site_id': str})
    print(f'  Candidate sites loaded: {len(sites)}')
    return sites


def load_well_metadata(sites_path):
    """Load NWIS well metadata, filtered to Upper Floridan."""
    if not os.path.exists(sites_path):
        print(f'ERROR: {sites_path} not found.')
        sys.exit(1)

    wells = pd.read_csv(sites_path, dtype={'site_no': str}, low_memory=False)
    print(f'  Total wells loaded: {len(wells)}')

    if 'nat_aqfr_cd' in wells.columns:
        floridan = wells[wells['nat_aqfr_cd'] == FLORIDAN_CODE].copy()
        print(f'  Upper Floridan wells: {len(floridan)}')
        if len(floridan) == 0:
            print('  WARNING: No Floridan wells. Using all wells.')
            floridan = wells.copy()
    else:
        print('  WARNING: nat_aqfr_cd column missing. Using all wells.')
        floridan = wells.copy()

    for col in ['dec_lat_va', 'dec_long_va']:
        floridan[col] = pd.to_numeric(floridan[col], errors='coerce')
    floridan = floridan.dropna(subset=['dec_lat_va', 'dec_long_va'])
    print(f'  Wells with coordinates: {len(floridan)}')
    return floridan


def parse_levels(levels_path):
    """
    Parse the NWIS levels CSV into a clean DataFrame with columns:
        site_no (str), date (datetime), depth_ft (float)

    Handles both:
      (a) Multi-index tuple format from dataretrieval.get_dv() saved with
          to_csv() — first column contains tuples like
          "('315921084242501', Timestamp('1990-01-01 00:00:00'))"
      (b) Flat format with separate site_no and date columns
    """
    if not os.path.exists(levels_path):
        print(f'ERROR: {levels_path} not found.')
        sys.exit(1)

    levels = pd.read_csv(levels_path, low_memory=False)
    print(f'  Levels records: {len(levels)}')

    # Identify depth column (NWIS parameter 72019 = depth-to-water, ft)
    depth_col = None
    for c in ['72019_Mean', '72019_mean', '72019']:
        if c in levels.columns:
            depth_col = c
            break
    if depth_col is None:
        for c in levels.columns:
            if '72019' in str(c):
                depth_col = c
                break
    if depth_col is None:
        raise RuntimeError(
            f'No depth-to-water column (NWIS 72019) found. '
            f'Columns: {list(levels.columns)[:20]}'
        )
    print(f'  Depth column: {depth_col}')

    # Detect format from the first column's first value
    first_col = levels.columns[0]
    first_val = str(levels[first_col].iloc[0])

    if first_val.startswith("(") and "Timestamp" in first_val:
        # Multi-index tuple format from dataretrieval.get_dv() to_csv()
        print(f'  Detected multi-index tuple format in column: {first_col}')
        tup = levels[first_col].astype(str)
        site_extracted = tup.str.extract(r"^\('?(\d+)'?", expand=False)
        date_extracted = tup.str.extract(r"Timestamp\('([^']+)'", expand=False)
        clean = pd.DataFrame({
            'site_no': site_extracted,
            'date': pd.to_datetime(date_extracted, errors='coerce'),
            'depth_ft': pd.to_numeric(levels[depth_col], errors='coerce'),
        })
    else:
        # Flat format: find separate site_no and date columns
        site_col = None
        for candidate in ['site_no', 'site_no_0']:
            if candidate in levels.columns:
                non_null = levels[candidate].notna().sum()
                if non_null > len(levels) * 0.5:
                    site_col = candidate
                    break
        if site_col is None:
            site_col = first_col
            print(f'  WARNING: falling back to first column for site_no: {site_col}')

        date_col = None
        for c in levels.columns:
            if 'date' in c.lower() or 'datetime' in c.lower():
                date_col = c
                break
        if date_col is None:
            raise RuntimeError(
                'No date column found in flat-format levels file. '
                'Trend computation requires per-record dates.'
            )
        print(f'  Site column: {site_col}, date column: {date_col}')

        clean = pd.DataFrame({
            'site_no': levels[site_col].astype(str),
            'date': pd.to_datetime(levels[date_col], errors='coerce'),
            'depth_ft': pd.to_numeric(levels[depth_col], errors='coerce'),
        })

    n_before = len(clean)
    clean = clean.dropna(subset=['site_no', 'date', 'depth_ft'])
    n_after = len(clean)
    if n_before != n_after:
        print(f'  Dropped {n_before - n_after} rows with missing site/date/depth')
    print(f'  Clean depth observations: {n_after}')
    print(f'  Unique wells in levels file: {clean["site_no"].nunique()}')

    return clean


def compute_per_well_metrics(levels_clean):
    """
    Compute per-well metrics:
      site_no, n_obs, n_years_distinct, mean_depth, depth_std, depth_slope_ft_yr
    Filters wells with fewer than MIN_OBS_COUNT observations.
    """
    rows = []
    n_skipped_obs = 0
    n_skipped_trend = 0

    for site_no, group in levels_clean.groupby('site_no'):
        n_obs = len(group)
        if n_obs < MIN_OBS_COUNT:
            n_skipped_obs += 1
            continue

        mean_depth = float(group['depth_ft'].mean())
        depth_std = float(group['depth_ft'].std())
        n_years_distinct = int(group['date'].dt.year.nunique())

        slope = np.nan
        if n_years_distinct >= MIN_TREND_YEARS:
            try:
                years = (
                    group['date'].dt.year.astype(float)
                    + (group['date'].dt.dayofyear.astype(float) - 1) / 365.25
                )
                fit = np.polyfit(years.values, group['depth_ft'].values, 1)
                slope = float(fit[0])
            except (np.linalg.LinAlgError, ValueError, TypeError):
                slope = np.nan
        if pd.isna(slope):
            n_skipped_trend += 1

        rows.append({
            'site_no': str(site_no),
            'n_obs': n_obs,
            'n_years_distinct': n_years_distinct,
            'mean_depth': round(mean_depth, 3),
            'depth_std': round(depth_std, 3) if not pd.isna(depth_std) else np.nan,
            'depth_slope_ft_yr': round(slope, 4) if not pd.isna(slope) else np.nan,
        })

    if n_skipped_obs > 0:
        print(f'  Skipped {n_skipped_obs} wells with < {MIN_OBS_COUNT} observations')
    if n_skipped_trend > 0:
        print(f'  {n_skipped_trend} wells lack sufficient data for trend (slope=NaN)')

    metrics = pd.DataFrame(rows)
    print(f'  Wells with usable metrics: {len(metrics)}')
    return metrics


def nearest_well_join(sites, wells, well_metrics):
    """
    For each candidate site, find the nearest Upper Floridan well and
    assign its sub-scores. No well within MAX_WELL_DIST_M -> default Ni.
    """
    wells_with_metrics = wells.merge(well_metrics, on='site_no', how='inner')
    print(f'  Wells with both coordinates and metrics: '
          f'{len(wells_with_metrics)}')

    if len(wells_with_metrics) == 0:
        print('  WARNING: no wells have both coordinates and metrics. '
              'All sites will use default Ni.')

    results = []
    for _, site in sites.iterrows():
        slat = float(site['latitude'])
        slon = float(site['longitude'])

        best_dist = float('inf')
        best_well = None
        for _, well in wells_with_metrics.iterrows():
            d = haversine_m(
                slat, slon,
                float(well['dec_lat_va']),
                float(well['dec_long_va']),
            )
            if d < best_dist:
                best_dist = d
                best_well = well

        if best_dist <= MAX_WELL_DIST_M and best_well is not None:
            mean_depth = best_well['mean_depth']
            slope = best_well['depth_slope_ft_yr']
            depth_std = best_well['depth_std']
            Ni, s_score, t_score, b_score = compose_Ni(mean_depth, slope, depth_std)

            results.append({
                'site_id': site['site_id'],
                'latitude': slat,
                'longitude': slon,
                'nearest_well_site_no': str(best_well['site_no']),
                'nearest_well_dist_m': round(best_dist, 1),
                'well_mean_depth_ft': round(float(mean_depth), 3),
                'well_depth_std_ft': (
                    round(float(depth_std), 3)
                    if not pd.isna(depth_std) else np.nan
                ),
                'well_depth_slope_ft_yr': (
                    round(float(slope), 4)
                    if not pd.isna(slope) else np.nan
                ),
                'well_n_obs': int(best_well['n_obs']),
                'well_n_years': int(best_well['n_years_distinct']),
                'storage_score': round(s_score, 4),
                'trend_score': round(t_score, 4),
                'responsiveness_bonus': round(b_score, 4),
                'Ni': round(Ni, 4),
            })
        else:
            results.append({
                'site_id': site['site_id'],
                'latitude': slat,
                'longitude': slon,
                'nearest_well_site_no': None,
                'nearest_well_dist_m': np.nan,
                'well_mean_depth_ft': np.nan,
                'well_depth_std_ft': np.nan,
                'well_depth_slope_ft_yr': np.nan,
                'well_n_obs': 0,
                'well_n_years': 0,
                'storage_score': np.nan,
                'trend_score': np.nan,
                'responsiveness_bonus': np.nan,
                'Ni': DEFAULT_SCORE,
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='NWIS storage capacity scoring with trend and responsiveness components.'
    )
    parser.add_argument(
        '--grid',
        type=int,
        choices=[20, 50],
        default=20,
        help='Grid size to process (20 or 50; default 20).',
    )
    args = parser.parse_args()

    sites_path, levels_path, grid_path, out_path = get_paths(args.grid)

    print('=' * 60)
    print(f'NWIS Storage Capacity — grid={args.grid}')
    print(f'  Composite Ni = '
          f'{W_STORAGE}*storage + {W_TREND}*trend + responsiveness_bonus')
    print(f'  Trend: penalize positive slope above '
          f'{TREND_DECLINE_THRESHOLD_FT_YR} ft/yr decline')
    print(f'  Responsiveness: bonus saturates at '
          f'{RESPONSIVENESS_SATURATION_FT} ft std '
          f'(max +{RESPONSIVENESS_BONUS_MAX:.2f})')
    print('=' * 60)
    print()

    print('Loading candidate sites...')
    sites = load_sites(grid_path)
    print()

    print('Loading NWIS well metadata...')
    wells = load_well_metadata(sites_path)
    print()

    print('Parsing groundwater levels file (this may take a moment)...')
    levels_clean = parse_levels(levels_path)
    print()

    print('Computing per-well metrics (mean depth, std, trend slope)...')
    well_metrics = compute_per_well_metrics(levels_clean)
    print()

    print('Performing nearest-well spatial join...')
    result = nearest_well_join(sites, wells, well_metrics)
    print()

    # Verify Ni in [0, 1]
    if result['Ni'].min() < 0.0 or result['Ni'].max() > 1.0:
        raise ValueError(
            f'Ni out of [0, 1]: '
            f'[{result["Ni"].min():.4f}, {result["Ni"].max():.4f}]'
        )

    # Report
    print('Results:')
    print(f'  Sites scored: {len(result)}')
    print(f'  Ni range: [{result["Ni"].min():.3f}, {result["Ni"].max():.3f}]')
    print(f'  Ni mean:  {result["Ni"].mean():.3f}')

    n_default = (result['nearest_well_site_no'].isna()).sum()
    print(f'  Sites using default '
          f'(no well within {MAX_WELL_DIST_M/1000:.0f} km): {n_default}')

    valid = result.dropna(subset=['storage_score'])
    if len(valid) > 0:
        print(f'  Sub-score stats (sites with assigned wells, n={len(valid)}):')
        print(f'    storage_score:        '
              f'mean={valid["storage_score"].mean():.3f}, '
              f'range=[{valid["storage_score"].min():.3f}, '
              f'{valid["storage_score"].max():.3f}]')
        print(f'    trend_score:          '
              f'mean={valid["trend_score"].mean():.3f}, '
              f'range=[{valid["trend_score"].min():.3f}, '
              f'{valid["trend_score"].max():.3f}]')
        print(f'    responsiveness_bonus: '
              f'mean={valid["responsiveness_bonus"].mean():.3f}, '
              f'range=[{valid["responsiveness_bonus"].min():.3f}, '
              f'{valid["responsiveness_bonus"].max():.3f}]')

        slopes = valid['well_depth_slope_ft_yr'].dropna()
        if len(slopes) > 0:
            n_declining = int((slopes > 0).sum())
            n_stable = int((slopes.abs() < 0.05).sum())
            n_recovering = int((slopes < -0.05).sum())
            print(f'  Trend direction (assigned wells, n={len(slopes)}):')
            print(f'    Declining (slope > 0):       {n_declining}')
            print(f'    Stable (|slope| < 0.05):     {n_stable}')
            print(f'    Recovering (slope < -0.05):  {n_recovering}')

    # Save
    result.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')

    return result


if __name__ == '__main__':
    main()
