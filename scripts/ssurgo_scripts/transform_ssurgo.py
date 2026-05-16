"""
VQ-MAR — SSURGO Infiltration Score (Si)

Computes the per-site soil-infiltration suitability score from USDA NRCS
SSURGO soil property and water-table data. SSURGO is the highest-weighted
component of the QUBO benefit term (30% of wi).

Method:
  - Filter to surface-horizon components (hzdept_r = 0); fall back to the
    shallowest horizon per cokey when no surface horizon exists.
  - Compute three component-level sub-scores:
      ksat:    ksat_norm = clip(ksat_r / KSAT_NORM, 0, 1)
      hydgrp:  HYDGRP_SCORES[hydgrp_clean]  (A=1.0, B=0.7, C=0.4, D=0.1;
                                             dual-class groups default to 0.1)
      texture: TEXTURE_SCORES[normalized texcl label]
  - Base composite: s_base = W_KSAT * ksat_norm + W_HYDGRP * hydgrp_score
                          + W_TEXCL * texcl_score
  - Clogging modifier (clay + organic-matter penalty):
      clogging_factor = clip(1.0 - K_CLAY * claytotal_r - K_OM * om_r,
                             CLOGGING_FLOOR, 1.0)
      s_comp = s_base * clogging_factor
  - Aggregate components to mukey via comppct_r-weighted mean to produce Si.
  - Hard-exclusion flag (ei_ssurgo = 1) for sites with wtdepannmin
    < WT_DEPTH_THRESH (300 cm). NULL wtdepannmin is imputed from the
    nearest NWIS well's well_mean_depth_ft (converted to cm via FT_TO_CM)
    when nwis_scores_{grid}.csv is available; the wtdepannmin_source
    column records ssurgo / nwis_imputed / missing for traceability.

Cross-source dependency: this transform reads nwis_scores_{grid}.csv
from georgia/processed/nwis/, so transform_nwis.py must run first for
the chosen grid size.

Architecture note:
  site_mukeys are matched to candidate sites by (latitude, longitude),
  not by site_id. site_ids may overlap across grid sizes, so a
  coordinate-based merge avoids cross-grid collisions; the same matching
  strategy is used in transform_nlcd.py and transform_osm.py. The 20-site
  and 50-site outputs are consistent because GA_001 through GA_020
  occupy the same coordinates in both candidate grids.

Input/output paths are selected by the --grid flag:
  --grid 20 (default):
    Soil:    georgia/raw/ssurgo/ssurgo_soil_properties.csv
    Water:   georgia/raw/ssurgo/ssurgo_water_table.csv
    Texture: georgia/raw/ssurgo/ssurgo_texture.csv
    Mukey:   georgia/raw/ssurgo/site_mukeys_20.csv
    Grid:    georgia/raw/ssurgo/candidate_grid_20.csv
    NWIS:    georgia/processed/nwis/nwis_scores.csv (for imputation)
    Output:  georgia/processed/ssurgo/ssurgo_scores.csv

  --grid 50:
    Soil:    georgia/raw/ssurgo/ssurgo_soil_properties_50sites.csv
    Water:   georgia/raw/ssurgo/ssurgo_water_table_50sites.csv
    Texture: georgia/raw/ssurgo/ssurgo_texture_50sites.csv
    Mukey:   georgia/raw/ssurgo/site_mukeys_50sites.csv
    Grid:    georgia/raw/ssurgo/candidate_grid_50sites.csv
    NWIS:    georgia/processed/nwis/nwis_scores_50.csv
    Output:  georgia/processed/ssurgo/ssurgo_scores_50.csv

Usage:
    python scripts/ssurgo_scripts/transform_ssurgo.py              # 20-site
    python scripts/ssurgo_scripts/transform_ssurgo.py --grid 50    # 50-site
"""

import argparse
import re
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
RAW_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
NWIS_PROC_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'nwis')
OUT_DIR = os.path.join(BASE_DIR, 'georgia', 'processed', 'ssurgo')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — ksat and hydrologic-group sub-scores
# ---------------------------------------------------------------------------
KSAT_NORM = 50.0        # μm/sec — normalization ceiling for ksat_r
WT_DEPTH_THRESH = 300   # cm — exclusion threshold for wtdepannmin

HYDGRP_SCORES = {
    'A':   1.0,
    'B':   0.7,
    'C':   0.4,
    'D':   0.1,
    'A/D': 0.1,
    'B/D': 0.1,
    'C/D': 0.1,
}

# ---------------------------------------------------------------------------
# Constants — texture, clogging, and NWIS-imputation parameters
# ---------------------------------------------------------------------------
# 3-component composite weights for the base infiltration score
W_KSAT = 0.50
W_HYDGRP = 0.30
W_TEXCL = 0.20

# 14-class texture lookup. Higher score = better infiltration for MAR.
# Matched against SSURGO chtexturegrp.texdesc values (case-insensitive,
# with common modifier prefixes stripped — see normalize_texture_label).
TEXTURE_SCORES = {
    # Coarse — excellent infiltration
    'sand':              1.00,
    'fine sand':         0.95,
    'loamy sand':        0.90,
    'loamy fine sand':   0.85,
    # Medium — good infiltration
    'sandy loam':        0.80,
    'fine sandy loam':   0.75,
    'loam':              0.60,
    # Silty / fine-medium — moderate
    'silt loam':         0.50,
    'silt':              0.40,
    'sandy clay loam':   0.45,
    # Clay-rich — poor infiltration
    'clay loam':         0.30,
    'silty clay loam':   0.20,
    'sandy clay':        0.20,
    'silty clay':        0.10,
    'clay':              0.05,
}

# Clogging modifier parameters
# clogging_factor = clip(1.0 - K_CLAY*claytotal_r - K_OM*om_r,
#                         CLOGGING_FLOOR, 1.0)
#  clay=10%, om=1% → factor = 0.94 (light penalty)
#  clay=30%, om=3% → factor = 0.82 (moderate)
#  clay=50%, om=5% → factor = 0.70 (heavy)
K_CLAY = 0.005
K_OM = 0.010
CLOGGING_FLOOR = 0.5  # max 50% reduction

# Conversion factor for NWIS imputation (well_mean_depth_ft → cm)
FT_TO_CM = 30.48


def get_paths(grid_size):
    """Return all input/output paths for the given grid size."""
    if grid_size == 20:
        return {
            'soil': os.path.join(RAW_DIR, 'ssurgo_soil_properties.csv'),
            'water': os.path.join(RAW_DIR, 'ssurgo_water_table.csv'),
            'texture': os.path.join(RAW_DIR, 'ssurgo_texture.csv'),
            'mukey': os.path.join(RAW_DIR, 'site_mukeys_20.csv'),
            'grid': os.path.join(RAW_DIR, 'candidate_grid_20.csv'),
            'nwis': os.path.join(NWIS_PROC_DIR, 'nwis_scores.csv'),
            'out': os.path.join(OUT_DIR, 'ssurgo_scores.csv'),
        }
    elif grid_size == 50:
        # Try _50sites variants first; fall back to base files.
        def first_existing(*candidates):
            for c in candidates:
                if os.path.exists(c):
                    return c
            return candidates[0]
        return {
            'soil': first_existing(
                os.path.join(RAW_DIR, 'ssurgo_soil_properties_50sites.csv'),
                os.path.join(RAW_DIR, 'ssurgo_soil_properties.csv'),
            ),
            'water': first_existing(
                os.path.join(RAW_DIR, 'ssurgo_water_table_50sites.csv'),
                os.path.join(RAW_DIR, 'ssurgo_water_table.csv'),
            ),
            'texture': first_existing(
                os.path.join(RAW_DIR, 'ssurgo_texture_50sites.csv'),
                os.path.join(RAW_DIR, 'ssurgo_texture.csv'),
            ),
            'mukey': first_existing(
                os.path.join(RAW_DIR, 'site_mukeys_50sites.csv'),
                os.path.join(RAW_DIR, 'site_mukeys_50.csv'),
            ),
            'grid': first_existing(
                os.path.join(RAW_DIR, 'candidate_grid_50sites.csv'),
                os.path.join(RAW_DIR, 'candidate_grid_50.csv'),
            ),
            'nwis': os.path.join(NWIS_PROC_DIR, 'nwis_scores_50.csv'),
            'out': os.path.join(OUT_DIR, 'ssurgo_scores_50.csv'),
        }
    else:
        raise ValueError(f'Unsupported grid size: {grid_size}. Use 20 or 50.')


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------
def normalize_texture_label(label):
    """Normalize a texture description string for dict lookup."""
    if pd.isna(label):
        return None
    s = str(label).strip().lower()
    # Strip common modifier prefixes (very, gravelly, stony, etc.)
    s = re.sub(r'^(very |extremely |gravelly |stony |cobbly |channery |bouldery |flaggy )+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None


def lookup_texture_score(label):
    """Look up texture score from a description, with substring fallback."""
    norm = normalize_texture_label(label)
    if norm is None:
        return np.nan
    if norm in TEXTURE_SCORES:
        return TEXTURE_SCORES[norm]
    # Substring match for compound names like "stratified sand to loam"
    for key, score in TEXTURE_SCORES.items():
        if key in norm:
            return score
    return np.nan


def load_texture(texture_path):
    """
    Load and process the SSURGO texture file. Returns a tuple
    (data_dict, status_string) where data_dict has 'scores' and 'labels'
    keyed by cokey.

    Defensively tries multiple texture column names.
    """
    if not os.path.exists(texture_path):
        print(f'  WARNING: {os.path.basename(texture_path)} not found. '
              f'Texture sub-score will default to neutral 0.5.')
        return None, 'missing_file'

    tex = pd.read_csv(texture_path, dtype={'cokey': str, 'chkey': str},
                      low_memory=False)
    print(f'  Loaded {len(tex)} texture records')
    print(f'  Available columns: {list(tex.columns)[:15]}')

    if 'cokey' not in tex.columns:
        print(f'  WARNING: No cokey column. Cannot join to soil components.')
        return None, 'missing_cokey'

    # Try multiple texture column names in priority order
    tex_col = None
    for candidate in ['texdesc', 'texture', 'texcl', 'texname',
                      'tex_class', 'texture_class', 'texdesc_h']:
        if candidate in tex.columns:
            tex_col = candidate
            break

    if tex_col is None:
        print(f'  WARNING: No texture column found in '
              f'{os.path.basename(texture_path)}. '
              f'Texture sub-score will default to 0.5 for all components.')
        return None, 'missing_texture_column'

    print(f'  Using texture column: {tex_col}')

    # Filter to surface horizon if hzdept_r exists
    if 'hzdept_r' in tex.columns:
        tex['hzdept_r'] = pd.to_numeric(tex['hzdept_r'], errors='coerce')
        tex = tex.sort_values(['cokey', 'hzdept_r'])
    elif 'rvindicator' in tex.columns:
        tex_rv = tex[tex['rvindicator'].astype(str).str.lower() == 'yes']
        if len(tex_rv) > 0:
            tex = tex_rv

    # Take the first texture per cokey (surface or representative)
    tex_first = tex.drop_duplicates(subset='cokey', keep='first').copy()
    tex_first['texture_score'] = tex_first[tex_col].apply(lookup_texture_score)
    n_scored = int(tex_first['texture_score'].notna().sum())
    print(f'  Texture scores assigned: {n_scored}/{len(tex_first)} cokeys')

    return {
        'scores': dict(zip(tex_first['cokey'].astype(str),
                            tex_first['texture_score'])),
        'labels': dict(zip(tex_first['cokey'].astype(str),
                            tex_first[tex_col])),
    }, 'ok'


# ---------------------------------------------------------------------------
# Site→mukey loading (lat/lon merge approach, matches NLCD/OSM)
# ---------------------------------------------------------------------------
def load_mukey_links(mukey_path, grid_path):
    """
    Load site→mukey links and filter to the candidate grid via lat/lon
    merge. This is the same fix used in NLCD/OSM for the cross-grid
    site_id collision bug.
    """
    if not os.path.exists(mukey_path):
        print(f'ERROR: {mukey_path} not found.')
        sys.exit(1)
    site_mukeys = pd.read_csv(mukey_path, dtype={'site_id': str, 'mukey': str})
    print(f'  Loaded {len(site_mukeys)} site→mukey records from '
          f'{os.path.basename(mukey_path)}')

    if not os.path.exists(grid_path):
        print(f'  WARNING: {grid_path} not found. Using all site_mukeys rows.')
        return site_mukeys.drop_duplicates(subset='site_id', keep='first')

    grid = pd.read_csv(grid_path, dtype={'site_id': str})

    if ('latitude' in site_mukeys.columns
            and 'longitude' in site_mukeys.columns
            and 'latitude' in grid.columns
            and 'longitude' in grid.columns):
        for col in ['latitude', 'longitude']:
            grid[col] = grid[col].round(6)
            site_mukeys[col] = site_mukeys[col].round(6)
        merged = grid[['site_id', 'latitude', 'longitude']].merge(
            site_mukeys.drop(columns=['site_id']),
            on=['latitude', 'longitude'],
            how='left',
        )
        merged = merged.drop_duplicates(subset='site_id', keep='first')
        print(f'  Filtered to {len(grid)}-site grid via lat/lon merge: '
              f'{len(merged)} rows')
        return merged
    else:
        # Fall back to site_id filter
        filtered = site_mukeys[
            site_mukeys['site_id'].isin(grid['site_id'])
        ].drop_duplicates(subset='site_id', keep='first')
        print(f'  Filtered to {len(grid)}-site grid via site_id: '
              f'{len(filtered)} rows')
        return filtered


# ---------------------------------------------------------------------------
# Component scoring (3-component + clogging)
# ---------------------------------------------------------------------------
def compute_component_scores_v2(soil, texture_data):
    """
    Component scoring:
        s_base = 0.50*ksat_norm + 0.30*hydgrp_score + 0.20*texcl_score
        s_comp = s_base * clogging_factor
    """
    soil['comppct_r'] = pd.to_numeric(soil['comppct_r'], errors='coerce')
    n_before = len(soil)
    soil = soil[soil['comppct_r'].fillna(0) >= 15].copy()
    if n_before - len(soil) > 0:
        print(f'  Filtered {n_before - len(soil)} minor components (<15%)')

    soil['hzdept_r'] = pd.to_numeric(soil['hzdept_r'], errors='coerce')
    surface = soil[soil['hzdept_r'] == 0].copy()
    print(f'  Surface horizon rows (hzdept_r=0): {len(surface)}')

    if surface.empty:
        print('  WARNING: No hzdept_r==0 rows. Using shallowest per cokey.')
        soil_sorted = soil.sort_values(['cokey', 'hzdept_r'])
        surface = soil_sorted.drop_duplicates(subset='cokey', keep='first').copy()

    surface['cokey'] = surface['cokey'].astype(str)

    # 1. ksat sub-score
    surface['ksat_r'] = pd.to_numeric(surface['ksat_r'], errors='coerce')
    surface['ksat_norm'] = (surface['ksat_r'] / KSAT_NORM).clip(0, 1).fillna(0.0)

    # 2. hydgrp sub-score
    surface['hydgrp_clean'] = surface['hydgrp'].astype(str).str.strip()
    surface['hydgrp_score'] = surface['hydgrp_clean'].map(HYDGRP_SCORES).fillna(0.4)

    # 3. Texture sub-score
    if texture_data is not None and 'scores' in texture_data:
        surface['texcl_score'] = surface['cokey'].map(texture_data['scores'])
        surface['texture_label'] = surface['cokey'].map(texture_data['labels'])
    else:
        surface['texcl_score'] = np.nan
        surface['texture_label'] = None
    n_missing_tex = int(surface['texcl_score'].isna().sum())
    if n_missing_tex > 0:
        print(f'  {n_missing_tex} components missing texture, defaulting to 0.5')
    surface['texcl_score'] = surface['texcl_score'].fillna(0.5)

    # 4. 3-component base score
    surface['s_base'] = (
        W_KSAT * surface['ksat_norm']
        + W_HYDGRP * surface['hydgrp_score']
        + W_TEXCL * surface['texcl_score']
    ).clip(0, 1)

    # 5. Clogging modifier
    if 'claytotal_r' in surface.columns:
        clay = pd.to_numeric(surface['claytotal_r'], errors='coerce')
    else:
        clay = pd.Series(np.nan, index=surface.index)
    if 'om_r' in surface.columns:
        om = pd.to_numeric(surface['om_r'], errors='coerce')
    else:
        om = pd.Series(np.nan, index=surface.index)
    surface['claytotal_r'] = clay
    surface['om_r'] = om

    if clay.notna().any() or om.notna().any():
        surface['clogging_factor'] = (
            1.0 - K_CLAY * clay.fillna(0.0) - K_OM * om.fillna(0.0)
        ).clip(CLOGGING_FLOOR, 1.0)
    else:
        print('  WARNING: claytotal_r and om_r both missing/empty. '
              'Clogging modifier disabled (factor=1.0).')
        surface['clogging_factor'] = 1.0

    surface['s_comp'] = (surface['s_base'] * surface['clogging_factor']).clip(0, 1)

    print(f'  Components scored: {len(surface)}')
    print(f'  s_base range:    [{surface["s_base"].min():.3f}, '
          f'{surface["s_base"].max():.3f}]')
    print(f'  clogging factor: mean={surface["clogging_factor"].mean():.3f}, '
          f'range=[{surface["clogging_factor"].min():.3f}, '
          f'{surface["clogging_factor"].max():.3f}]')
    print(f'  s_comp range:    [{surface["s_comp"].min():.3f}, '
          f'{surface["s_comp"].max():.3f}]')

    keep = ['mukey', 'cokey', 'comppct_r', 's_base', 's_comp',
            'ksat_norm', 'hydgrp_clean', 'hydgrp_score',
            'texcl_score', 'texture_label',
            'claytotal_r', 'om_r', 'clogging_factor']
    return surface[keep].copy()


def aggregate_to_mukey(comp_scores):
    """Comppct_r-weighted aggregation of component scores to mukey level."""
    def w_mean(group, col):
        weights = group['comppct_r'].fillna(0)
        if weights.sum() == 0:
            return group[col].mean()
        return float(np.average(group[col], weights=weights))

    rows = []
    for mukey, group in comp_scores.groupby('mukey'):
        rows.append({
            'mukey': str(mukey),
            'Si_3component': w_mean(group, 's_base'),
            'Si': w_mean(group, 's_comp'),
            'clogging_factor_mu': w_mean(group, 'clogging_factor'),
            'ksat_norm_mu': w_mean(group, 'ksat_norm'),
            'hydgrp_score_mu': w_mean(group, 'hydgrp_score'),
            'texcl_score_mu': w_mean(group, 'texcl_score'),
        })
    out = pd.DataFrame(rows)
    out['Si'] = out['Si'].clip(0, 1)
    out['Si_3component'] = out['Si_3component'].clip(0, 1)
    return out


# ---------------------------------------------------------------------------
# NWIS imputation (cross-source dependency)
# ---------------------------------------------------------------------------
def impute_wtdepannmin_from_nwis(sites_with_wt, nwis_path):
    """
    For sites where wtdepannmin is NULL, impute from the nearest NWIS well's
    well_mean_depth_ft (converted to cm). Reads nwis_scores_{grid}.csv
    produced by transform_nwis.py.
    """
    sites = sites_with_wt.copy()
    sites['wtdepannmin_source'] = np.where(
        sites['wtdepannmin'].notna(), 'ssurgo', 'missing'
    )

    if not os.path.exists(nwis_path):
        print(f'  WARNING: {os.path.basename(nwis_path)} not found. '
              f'Cannot impute — NULL wtdepannmin values will remain NULL.')
        return sites

    nwis = pd.read_csv(nwis_path, dtype={'site_id': str})
    if 'well_mean_depth_ft' not in nwis.columns:
        print(f'  WARNING: nwis_scores file lacks well_mean_depth_ft. '
              f'Cannot impute.')
        return sites

    nwis_lookup = nwis.set_index('site_id')['well_mean_depth_ft'].to_dict()

    n_imputed = 0
    n_unimputable = 0
    for idx, row in sites.iterrows():
        if pd.isna(row['wtdepannmin']):
            site_id = row['site_id']
            nwis_depth_ft = nwis_lookup.get(site_id, np.nan)
            if pd.notna(nwis_depth_ft):
                sites.at[idx, 'wtdepannmin'] = float(nwis_depth_ft) * FT_TO_CM
                sites.at[idx, 'wtdepannmin_source'] = 'nwis_imputed'
                n_imputed += 1
            else:
                n_unimputable += 1

    print(f'  Imputed {n_imputed} sites from NWIS, '
          f'{n_unimputable} sites still missing wtdepannmin')
    return sites


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='SSURGO infiltration scoring with clogging modifier and NWIS imputation.'
    )
    parser.add_argument(
        '--grid',
        type=int,
        choices=[20, 50],
        default=20,
        help='Grid size to process (20 or 50; default 20).',
    )
    args = parser.parse_args()
    paths = get_paths(args.grid)

    print('=' * 60)
    print(f'SSURGO Infiltration Score — grid={args.grid}')
    print(f'  Si_base = '
          f'{W_KSAT}*ksat + {W_HYDGRP}*hydgrp + {W_TEXCL}*texcl')
    print('  Si = Si_base * clogging_factor (clay + organic matter)')
    print('  NULL wtdepannmin imputed from nearest NWIS well')
    print('=' * 60)
    print()

    print('Loading raw SSURGO data...')
    if not os.path.exists(paths['soil']):
        print(f'ERROR: {paths["soil"]} not found.')
        sys.exit(1)
    soil = pd.read_csv(
        paths['soil'],
        dtype={'mukey': str, 'cokey': str, 'chkey': str},
        low_memory=False,
    )
    print(f'  Soil properties: {len(soil)} rows from '
          f'{os.path.basename(paths["soil"])}')

    if not os.path.exists(paths['water']):
        print(f'ERROR: {paths["water"]} not found.')
        sys.exit(1)
    water_table = pd.read_csv(paths['water'], dtype={'mukey': str})
    water_table['wtdepannmin'] = pd.to_numeric(
        water_table['wtdepannmin'], errors='coerce'
    )
    water_table = water_table[['mukey', 'wtdepannmin']].copy()
    print(f'  Water table: {len(water_table)} rows')
    print()

    print('Loading texture data...')
    texture_data, tex_status = load_texture(paths['texture'])
    print(f'  Texture status: {tex_status}')
    print()

    print('Loading site→mukey links...')
    site_mukeys = load_mukey_links(paths['mukey'], paths['grid'])
    print()

    print('Computing component scores (3-component + clogging)...')
    comp_scores = compute_component_scores_v2(soil, texture_data)
    print()

    print('Aggregating component scores to mukey...')
    mukey_scores = aggregate_to_mukey(comp_scores)
    print(f'  Mukeys scored: {len(mukey_scores)}')
    print(f'  Si range: [{mukey_scores["Si"].min():.3f}, '
          f'{mukey_scores["Si"].max():.3f}]')
    print()

    print('Joining mukey scores and water table to sites...')
    result = site_mukeys.merge(mukey_scores, on='mukey', how='left')
    result = result.merge(water_table, on='mukey', how='left')
    result['Si'] = result['Si'].fillna(0.0)
    result['Si_3component'] = result['Si_3component'].fillna(0.0)
    print()

    print('Imputing NULL wtdepannmin from NWIS...')
    result = impute_wtdepannmin_from_nwis(result, paths['nwis'])
    print()

    print('Computing exclusion flags (after imputation)...')
    excluded_mask = (
        result['wtdepannmin'].notna()
        & (result['wtdepannmin'] < WT_DEPTH_THRESH)
    )
    result['ei_ssurgo'] = excluded_mask.astype(int)
    print(f'  Sites excluded by wtdepannmin < {WT_DEPTH_THRESH} cm: '
          f'{int(result["ei_ssurgo"].sum())}')
    print()

    # Verify Si in [0, 1]
    if result['Si'].min() < 0.0 or result['Si'].max() > 1.0:
        raise ValueError(
            f'Si out of [0, 1]: '
            f'[{result["Si"].min():.4f}, {result["Si"].max():.4f}]'
        )

    # Build output schema
    out_cols = [
        'site_id', 'latitude', 'longitude', 'mukey',
        'ksat_norm_mu', 'hydgrp_score_mu', 'texcl_score_mu',
        'clogging_factor_mu',
        'Si_3component', 'Si',
        'wtdepannmin', 'wtdepannmin_source',
        'ei_ssurgo',
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    final = result[out_cols].copy()

    final.to_csv(paths['out'], index=False)
    print(f'Saved to {paths["out"]}')
    print()

    # Report
    print('Results:')
    print(f'  Sites scored:        {len(final)}')
    print(f'  Si range:            [{final["Si"].min():.3f}, '
          f'{final["Si"].max():.3f}]')
    print(f'  Si mean:             {final["Si"].mean():.3f}')
    if 'Si_3component' in final.columns:
        delta = (final['Si_3component'] - final['Si']).mean()
        print(f'  Mean clogging penalty: {delta:.4f} '
              f'(Si_3component minus Si)')
    print(f'  Sites excluded (ei_ssurgo=1): {int(final["ei_ssurgo"].sum())}')
    if 'wtdepannmin_source' in final.columns:
        print(f'  wtdepannmin sources:')
        for source, count in final['wtdepannmin_source'].value_counts().items():
            print(f'    {source}: {count}')

    return final


if __name__ == '__main__':
    main()
