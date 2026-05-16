"""
VQ-MAR — Site Scoring (Flat Cost Model)

Composite benefit:
  wi = 0.30*Si + 0.25*Ni + 0.15*Li + 0.15*Cclim,i + 0.15*Ai
  All wi verified in [0, 1].
  Flat Ci = 0.5 for all sites.

Joins the five processed score files on site_id, computes the
composite wi, and exports the 20-site scored file for QUBO assembly.

Input:  georgia/processed/{ssurgo,nwis,nlcd,noaa,osm}/*_scores.csv
Output: georgia/processed/sites_20_scored.csv
        georgia/qiskit-ready/site_metadata/sites_20_scored.csv (copy)
"""

import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)
PROC_DIR = os.path.join(BASE_DIR, 'georgia', 'processed')
RAW_SSURGO = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
QISKIT_DIR = os.path.join(BASE_DIR, 'georgia', 'qiskit-ready')
os.makedirs(os.path.join(QISKIT_DIR, 'site_metadata'), exist_ok=True)

# ---------------------------------------------------------------------------
# Weights  (Data Spec §2)
# ---------------------------------------------------------------------------
W_SSURGO = 0.30   # Infiltration Si
W_NWIS   = 0.25   # Storage capacity Ni
W_NLCD   = 0.15   # Land-use suitability Li
W_NOAA   = 0.15   # Climate priority Cclim,i
W_OSM    = 0.15   # Infrastructure access Ai

# Flat cost model
FLAT_CI = 0.5


def load_scores():
    """Load all five processed score files."""
    paths = {
        'ssurgo': os.path.join(PROC_DIR, 'ssurgo', 'ssurgo_scores.csv'),
        'nwis':   os.path.join(PROC_DIR, 'nwis', 'nwis_scores.csv'),
        'nlcd':   os.path.join(PROC_DIR, 'nlcd', 'nlcd_scores.csv'),
        'noaa':   os.path.join(PROC_DIR, 'noaa', 'noaa_scores.csv'),
        'osm':    os.path.join(PROC_DIR, 'osm', 'osm_scores.csv'),
    }

    dfs = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f'ERROR: {path} not found.')
            print(f'  Run transform_{name}.py first.')
            sys.exit(1)
        dfs[name] = pd.read_csv(path, dtype={'site_id': str})
        print(f'  {name:8s}: {len(dfs[name]):3d} sites loaded')

    return dfs


def join_scores(dfs):
    """
    Left-join all score files onto the SSURGO base (which has
    the canonical site_id list for the 20-site grid).
    """
    # Start from the candidate grid for a clean base
    grid_20_path = os.path.join(RAW_SSURGO, 'candidate_grid_20.csv')
    if os.path.exists(grid_20_path):
        base = pd.read_csv(grid_20_path, dtype={'site_id': str})
    else:
        # Fall back to SSURGO scores as the base
        base = dfs['ssurgo'][['site_id', 'latitude', 'longitude']].copy()

    # Join SSURGO scores
    ssurgo_cols = dfs['ssurgo'][['site_id', 'Si', 'ei_ssurgo']].copy()
    result = base.merge(ssurgo_cols, on='site_id', how='left')

    # Join NWIS scores
    nwis_cols = dfs['nwis'][['site_id', 'Ni']].copy()
    result = result.merge(nwis_cols, on='site_id', how='left')

    # Join NLCD scores
    nlcd_cols = dfs['nlcd'][['site_id', 'Li', 'ei_nlcd']].copy()
    result = result.merge(nlcd_cols, on='site_id', how='left')

    # Join NOAA scores
    noaa_cols = dfs['noaa'][['site_id', 'Cclim_i']].copy()
    result = result.merge(noaa_cols, on='site_id', how='left')

    # Join OSM scores
    osm_cols = dfs['osm'][['site_id', 'Ai']].copy()
    result = result.merge(osm_cols, on='site_id', how='left')

    return result


def compute_wi(df):
    """
    Composite benefit: wi = 0.30*Si + 0.25*Ni + 0.15*Li + 0.15*Cclim,i + 0.15*Ai

    Fill missing sub-scores with conservative defaults before computing.
    """
    # Default fill values for any missing scores
    df['Si'] = df['Si'].fillna(0.0)
    df['Ni'] = df['Ni'].fillna(0.5)
    df['Li'] = df['Li'].fillna(0.5)
    df['Cclim_i'] = df['Cclim_i'].fillna(0.65)
    df['Ai'] = df['Ai'].fillna(0.0)

    # Compute weighted composite
    df['wi'] = (
        W_SSURGO * df['Si']
        + W_NWIS * df['Ni']
        + W_NLCD * df['Li']
        + W_NOAA * df['Cclim_i']
        + W_OSM  * df['Ai']
    )

    # Flat cost
    df['Ci'] = FLAT_CI

    # Combined exclusion flag: ei = 1 if excluded by ANY source
    df['ei_ssurgo'] = df['ei_ssurgo'].fillna(0).astype(int)
    df['ei_nlcd'] = df['ei_nlcd'].fillna(0).astype(int)
    df['ei'] = ((df['ei_ssurgo'] == 1) | (df['ei_nlcd'] == 1)).astype(int)

    # ── Comprehensive bounds verification ──────────────────────────────
    # Every coefficient that feeds the QUBO must be in [0, 1].
    # Catching violations here prevents silent corruption of the
    # Q matrix during QUBO assembly.
    bounds_checks = {
        'Si': df['Si'], 'Ni': df['Ni'], 'Li': df['Li'],
        'Cclim_i': df['Cclim_i'], 'Ai': df['Ai'],
        'wi': df['wi'], 'Ci': df['Ci'],
    }
    for name, series in bounds_checks.items():
        lo, hi = series.min(), series.max()
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f'{name} out of [0, 1] range: [{lo:.6f}, {hi:.6f}]. '
                f'Check the upstream transform script.'
            )
    print('  All coefficients verified in [0, 1].')

    return df


def main():
    print('=' * 60)
    print('VQ-MAR — Site Scoring (Flat Cost Model)')
    print('  wi = 0.30*Si + 0.25*Ni + 0.15*Li + 0.15*Cclim,i + 0.15*Ai')
    print(f'  Flat Ci = {FLAT_CI} for all sites')
    print('=' * 60)
    print()

    # Step 1: Load all processed score files
    print('Loading processed score files...')
    dfs = load_scores()
    print()

    # Step 2: Join on site_id
    print('Joining scores on site_id...')
    joined = join_scores(dfs)
    print(f'  Joined dataset: {len(joined)} sites')
    # Guard against join-induced row duplication or loss
    if len(joined) != 20:
        print(f'  WARNING: Expected 20 sites, got {len(joined)}. '
              f'Check for duplicate site_ids in upstream score files.')
    print()

    # Step 3: Compute wi
    print('Computing wi...')
    result = compute_wi(joined)
    print(f'  wi range: [{result["wi"].min():.4f}, {result["wi"].max():.4f}]')
    print(f'  wi mean:  {result["wi"].mean():.4f}')
    print(f'  Excluded sites (ei=1): {result["ei"].sum()}')
    print()

    # Step 4: Save to processed/
    out_path = os.path.join(PROC_DIR, 'sites_20_scored.csv')
    result.to_csv(out_path, index=False)
    print(f'  Primary output:  {out_path}')

    # Also copy to qiskit-ready/ for downstream QUBO assembly
    qiskit_path = os.path.join(QISKIT_DIR, 'site_metadata', 'sites_20_scored.csv')
    result.to_csv(qiskit_path, index=False)
    print(f'  Qiskit copy:     {qiskit_path}')
    print()

    # Step 5: Summary table
    print('=' * 60)
    print('SITE SCORING SUMMARY (20 sites)')
    print('=' * 60)
    summary_cols = ['site_id', 'Si', 'Ni', 'Li', 'Cclim_i', 'Ai', 'wi', 'Ci', 'ei']
    print(result[summary_cols].to_string(index=False, float_format='%.3f'))
    print()

    # Sub-score statistics
    print('Sub-score statistics:')
    for col, weight, label in [
        ('Si', W_SSURGO, 'SSURGO infiltration'),
        ('Ni', W_NWIS, 'NWIS storage'),
        ('Li', W_NLCD, 'NLCD land-use'),
        ('Cclim_i', W_NOAA, 'NOAA climate'),
        ('Ai', W_OSM, 'OSM infrastructure'),
    ]:
        vals = result[col]
        print(f'  {label:25s} (w={weight:.2f}): '
              f'mean={vals.mean():.3f}, '
              f'min={vals.min():.3f}, max={vals.max():.3f}')
    print()

    print('Output: sites_20_scored.csv')
    print('  Contains: site_id, lat, lon, Si, Ni, Li, Cclim_i, Ai, wi, Ci, ei')

    return result


if __name__ == '__main__':
    main()
