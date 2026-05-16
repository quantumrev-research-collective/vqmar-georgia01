"""
VQ-MAR — NOAA Climate Priority (Cclim,i)

Computes the climate-priority sub-score (Cclim,i) for each candidate
site from NOAA daily climate data at the Albany SW Georgia Regional
Airport station (GHCND:USW00013869), the single weather station
serving the Dougherty Plain study area.

Method:
  - Daily PET (potential evapotranspiration) is computed via
    Hargreaves-Samani 1985: PET = 0.0023 * (Tmean + 17.8) *
    sqrt(Tmax - Tmin) * Ra, where Ra is extraterrestrial radiation
    at the station latitude.
  - Three sub-scores are computed from the multi-year time series:
      * Adequacy (40%):  annual P / annual PET ratio, clipped to [0, 1]
      * Seasonal (30%):  Walsh-Lawler PCI proximity to PCI_OPTIMAL
                         (tent function, width PCI_TOLERANCE)
      * Extremes (30%):  count of heavy-rainfall days per year,
                         saturating at EXTREME_DAYS_SATURATION
  - Composite: Cclim,i = 0.40 * adequacy + 0.30 * seasonal + 0.30 * extremes
  - Years 2000 and 2004 are excluded (PRCP gaps in the CDO API record);
    years with fewer than MIN_DAYS_PER_YEAR daily records are dropped.

Note: NOAA is single-station for the entire Dougherty Plain study area,
so Cclim,i is uniform across all sites in the grid. The diagnostics
columns (annual P, annual PET, Walsh-Lawler PCI, etc.) are written to
the output CSV alongside the score for traceability in the paper's
hydrologic validation section.

Input/output paths are selected by the --grid flag:
  --grid 20 (default):
    Climate: georgia/raw/noaa/noaa_daily_climate.csv
    Grid:    georgia/raw/ssurgo/candidate_grid_20.csv
    Output:  georgia/processed/noaa/noaa_scores.csv

  --grid 50:
    Climate: georgia/raw/noaa/noaa_daily_climate_50sites.csv
             (falls back to noaa_daily_climate.csv if not present;
              NOAA queries are single-station, so site count does not
              change the underlying climate record)
    Grid:    georgia/raw/ssurgo/candidate_grid_50sites.csv
    Output:  georgia/processed/noaa/noaa_scores_50.csv

Usage:
    python scripts/noaa_scripts/transform_noaa.py              # 20-site (default)
    python scripts/noaa_scripts/transform_noaa.py --grid 50    # 50-site
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
RAW_NOAA = os.path.join(BASE_DIR, 'georgia', 'raw', 'noaa')
RAW_SSURGO = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
OUT_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'noaa')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Climate computation constants
# ---------------------------------------------------------------------------
# Albany SW Georgia Regional Airport (GHCND:USW00013869)
STATION_LATITUDE = 31.5353

# Years to exclude from the climate aggregation (PRCP gaps from CDO API)
EXCLUDE_YEARS = {2000, 2004}

# Drop years with fewer than this many daily records (data completeness gate)
MIN_DAYS_PER_YEAR = 300

# Sub-score weights for Cclim,i
W_ADEQUACY = 0.40
W_SEASONAL = 0.30
W_EXTREMES = 0.30

# Walsh-Lawler PCI optimal value for MAR (moderate seasonality preferred).
# Score is a tent function peaking at PCI_OPTIMAL with width PCI_TOLERANCE.
# These are tunable — defaults reflect "moderate seasonality is best for MAR"
# (a wet season concentrates recharge; uniform monthly rainfall is less ideal).
PCI_OPTIMAL = 12.0
PCI_TOLERANCE = 10.0

# Threshold for "heavy rainfall day" in mm.
# 25 mm/day matches the WMO "heavy rain" classification.
EXTREME_DAY_THRESHOLD_MM = 25.0
# Saturation: extreme_score = 1.0 at 30 heavy-rainfall days/year and above.
EXTREME_DAYS_SATURATION = 30.0


def get_paths(grid_size):
    """Return (climate_path, grid_path, output_path) for the given grid size."""
    if grid_size == 20:
        return (
            os.path.join(RAW_NOAA, 'noaa_daily_climate.csv'),
            os.path.join(RAW_SSURGO, 'candidate_grid_20.csv'),
            os.path.join(OUT_DIR, 'noaa_scores.csv'),
        )
    elif grid_size == 50:
        # Try the _50sites variant first; fall back to the original.
        # NOAA queries are single-station, so site count does not change
        # the underlying climate record.
        path_50 = os.path.join(RAW_NOAA, 'noaa_daily_climate_50sites.csv')
        path_20 = os.path.join(RAW_NOAA, 'noaa_daily_climate.csv')
        climate_path = path_50 if os.path.exists(path_50) else path_20
        return (
            climate_path,
            os.path.join(RAW_SSURGO, 'candidate_grid_50sites.csv'),
            os.path.join(OUT_DIR, 'noaa_scores_50.csv'),
        )
    else:
        raise ValueError(f'Unsupported grid size: {grid_size}. Use 20 or 50.')


# ---------------------------------------------------------------------------
# Hargreaves-Samani PET helpers
# ---------------------------------------------------------------------------
def extraterrestrial_radiation(lat_deg, day_of_year):
    """
    Extraterrestrial radiation Ra (MJ/m²/day) per FAO-56 Eq. 21.
    Inputs:
      lat_deg:     latitude in decimal degrees
      day_of_year: integer 1..365 (or 366)
    """
    Gsc = 0.0820  # solar constant, MJ/m²/min
    phi = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2 * math.pi * day_of_year / 365.0)
    delta = 0.409 * math.sin(2 * math.pi * day_of_year / 365.0 - 1.39)
    # Sunset hour angle, clipped to handle high-latitude / extreme cases
    ws_arg = -math.tan(phi) * math.tan(delta)
    ws_arg = max(-1.0, min(1.0, ws_arg))
    ws = math.acos(ws_arg)
    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return Ra


def compute_daily_pet_mm(daily, station_lat):
    """
    Vectorized Hargreaves-Samani 1985 PET in mm/day:
      PET = 0.0023 × (Tmean + 17.8) × √(Tmax − Tmin) × Ra(mm/day)
    where Ra(mm/day) = 0.408 × Ra(MJ/m²/day) per FAO-56 conversion.
    """
    # Cache Ra by day-of-year to avoid recomputing per record
    ra_cache = {doy: extraterrestrial_radiation(station_lat, doy)
                for doy in range(1, 367)}
    ra_mj = daily['day_of_year'].map(ra_cache)
    ra_mm = 0.408 * ra_mj  # MJ/m²/day → mm/day water equivalent

    tmax = pd.to_numeric(daily['TMAX_C'], errors='coerce')
    tmin = pd.to_numeric(daily['TMIN_C'], errors='coerce')
    delta_t = tmax - tmin
    tmean = (tmax + tmin) / 2.0

    valid = tmax.notna() & tmin.notna() & (delta_t >= 0)
    pet = np.where(
        valid,
        0.0023 * (tmean + 17.8) * np.sqrt(delta_t.clip(lower=0)) * ra_mm,
        np.nan,
    )
    return pet, ra_mj


def compute_climate_score(daily, station_lat):
    """
    Aggregate the daily climate time series into a single Cclim_i value.
    Returns (Cclim_i, diagnostics_dict).
    """
    # Step 1: Parse dates and exclude bad-data years
    daily = daily.copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce')
    daily = daily.dropna(subset=['date'])
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    daily['day_of_year'] = daily['date'].dt.dayofyear

    n_before = len(daily)
    daily = daily[~daily['year'].isin(EXCLUDE_YEARS)].reset_index(drop=True)
    n_after = len(daily)
    print(f'  Excluded {n_before - n_after} records from years {sorted(EXCLUDE_YEARS)}')

    if len(daily) == 0:
        raise RuntimeError('No daily climate records remain after year exclusion.')

    # Step 2: Daily PET via Hargreaves-Samani (vectorized)
    daily['PET_mm'], daily['Ra'] = compute_daily_pet_mm(daily, station_lat)

    # Step 3: Annual aggregates, drop incomplete years
    annual = daily.groupby('year').agg(
        P_mm=('PRCP_mm', 'sum'),
        PET_mm=('PET_mm', 'sum'),
        n_days=('PRCP_mm', 'count'),
    ).reset_index()
    annual = annual[annual['n_days'] >= MIN_DAYS_PER_YEAR].reset_index(drop=True)
    n_years = len(annual)
    if n_years == 0:
        raise RuntimeError(
            f'No complete years remain (>= {MIN_DAYS_PER_YEAR} days) '
            f'for climate aggregation.'
        )

    mean_P = annual['P_mm'].mean()
    mean_PET = annual['PET_mm'].mean()
    print(f'  Years used: {n_years}')
    print(f'  Mean annual P:   {mean_P:.0f} mm')
    print(f'  Mean annual PET: {mean_PET:.0f} mm')

    # Step 4: Adequacy sub-score (40%) — clipped P/PET ratio
    p_pet_ratio = mean_P / mean_PET if mean_PET > 0 else 0.0
    adequacy_score = min(max(p_pet_ratio, 0.0), 1.0)
    print(f'  P/PET ratio:      {p_pet_ratio:.3f}')
    print(f'  Adequacy score:   {adequacy_score:.3f}  (weight {W_ADEQUACY})')

    # Step 5: Seasonal concentration sub-score (30%) — Walsh-Lawler PCI
    monthly = daily.groupby('month')['PRCP_mm'].sum()
    if monthly.sum() > 0:
        pci = 100.0 * (monthly ** 2).sum() / (monthly.sum() ** 2)
    else:
        pci = 8.3  # uniform fallback
    seasonal_score = max(0.0, 1.0 - abs(pci - PCI_OPTIMAL) / PCI_TOLERANCE)
    seasonal_score = min(seasonal_score, 1.0)
    print(f'  Walsh-Lawler PCI: {pci:.2f}  (optimal {PCI_OPTIMAL})')
    print(f'  Seasonal score:   {seasonal_score:.3f}  (weight {W_SEASONAL})')

    # Step 6: Extreme events sub-score (30%) — heavy-rainfall day frequency
    extreme_days = daily[daily['PRCP_mm'] >= EXTREME_DAY_THRESHOLD_MM]
    extreme_days_per_year = len(extreme_days) / n_years
    extreme_score = min(extreme_days_per_year / EXTREME_DAYS_SATURATION, 1.0)
    print(f'  Heavy-rain days (>={EXTREME_DAY_THRESHOLD_MM} mm)/yr: '
          f'{extreme_days_per_year:.1f}')
    print(f'  Extremes score:   {extreme_score:.3f}  (weight {W_EXTREMES})')

    # Step 7: Composite
    cclim = (
        W_ADEQUACY * adequacy_score
        + W_SEASONAL * seasonal_score
        + W_EXTREMES * extreme_score
    )
    cclim = min(max(cclim, 0.0), 1.0)
    print(f'\n  Cclim_i = {W_ADEQUACY}·{adequacy_score:.3f} '
          f'+ {W_SEASONAL}·{seasonal_score:.3f} '
          f'+ {W_EXTREMES}·{extreme_score:.3f} = {cclim:.4f}')

    diagnostics = {
        'n_years_used': n_years,
        'mean_P_mm': round(float(mean_P), 1),
        'mean_PET_mm': round(float(mean_PET), 1),
        'P_PET_ratio': round(float(p_pet_ratio), 3),
        'walsh_lawler_PCI': round(float(pci), 2),
        'extreme_days_per_year': round(float(extreme_days_per_year), 2),
        'adequacy_score': round(float(adequacy_score), 4),
        'seasonal_score': round(float(seasonal_score), 4),
        'extreme_score': round(float(extreme_score), 4),
    }
    return cclim, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description='NOAA climate priority scoring (PET-driven sub-scores).'
    )
    parser.add_argument(
        '--grid',
        type=int,
        choices=[20, 50],
        default=20,
        help='Grid size to process (20 or 50; default 20).',
    )
    args = parser.parse_args()

    climate_path, grid_path, out_path = get_paths(args.grid)

    print('=' * 60)
    print(f'NOAA Climate Priority — grid={args.grid}')
    print('  Hargreaves-Samani PET + 3 sub-scores')
    print(f'  Weights: adequacy={W_ADEQUACY}, seasonal={W_SEASONAL}, '
          f'extremes={W_EXTREMES}')
    print(f'  Excluding years: {sorted(EXCLUDE_YEARS)}')
    print('=' * 60)
    print()

    # Step 1: Load climate data
    if not os.path.exists(climate_path):
        print(f'ERROR: {climate_path} not found.')
        sys.exit(1)
    print(f'Loading climate data from {os.path.basename(climate_path)}...')
    daily = pd.read_csv(climate_path)
    print(f'  Loaded: {len(daily)} daily records')
    print()

    # Step 2: Compute Cclim_i (uniform across all sites because NOAA is
    # single-station for the entire Dougherty Plain study area)
    print('Computing climate score from station time series...')
    cclim, diagnostics = compute_climate_score(daily, STATION_LATITUDE)
    print()

    # Step 3: Load grid and assign uniform score
    if not os.path.exists(grid_path):
        print(f'ERROR: {grid_path} not found.')
        sys.exit(1)
    sites = pd.read_csv(grid_path, dtype={'site_id': str})
    print(f'  Candidate sites: {len(sites)}')

    result = sites[['site_id', 'latitude', 'longitude']].copy()
    result['Cclim_i'] = cclim

    # Add diagnostics columns (uniform across sites; useful for traceability
    # in the §VI hydrologic validation paragraph)
    for key, val in diagnostics.items():
        result[key] = val

    # Step 4: Verify Cclim_i in [0, 1]
    if result['Cclim_i'].min() < 0.0 or result['Cclim_i'].max() > 1.0:
        raise ValueError(
            f'Cclim_i out of [0, 1]: '
            f'[{result["Cclim_i"].min():.4f}, {result["Cclim_i"].max():.4f}]'
        )

    # Step 5: Save
    result.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')
    print(f'  All {len(result)} sites assigned Cclim_i = {cclim:.4f}')

    return result


if __name__ == '__main__':
    main()
