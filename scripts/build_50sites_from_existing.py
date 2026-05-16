"""
VQ-MAR — Build 50-Site Dataset from Existing Files

The fetch_ssurgo_spatial.py script resets the site_id counter for every
grid size, so both the 20-site and 50-site grids use site_ids GA_001+
at different coordinates. The downstream files (site_nlcd_classes.csv,
site_osm_distances.csv) dedupe by (latitude, longitude), so they contain
139 unique coordinates with ambiguous site_id values where the same
site_id can appear at multiple coordinates.

Filtering by site_id therefore returns rows from both grids that share
an ID — the wrong rows. This script filters by (latitude, longitude)
instead, and replaces the ambiguous site_id in the output with the
canonical site_id from the 50-site grid so downstream joins work
correctly.

NWIS, NOAA, and SSURGO are unaffected:
  - NWIS is county-scoped (copied wholesale).
  - NOAA is station-scoped (copied wholesale).
  - SSURGO is filtered by mukey, which is a soil unit identifier with no
    cross-grid collision.

Makes no API calls and creates no new data — pure filtering and patching
of existing files.

Usage:
    python scripts/build_50sites_from_existing.py
"""

import pandas as pd
import shutil
import os

# Project paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

SSURGO_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
NLCD_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'nlcd')
NWIS_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'nwis')
NOAA_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'noaa')
OSM_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'osm')

# OSM silent-failure patch parameters
PATCH_VALUE = 10000.0
DISTANCE_FIELDS = ['dist_road_m', 'dist_waterway_m', 'dist_water_infra_m']

# Lat/lon precision for safe merging (matches fetch_ssurgo_spatial.py round)
COORD_PRECISION = 6


def merge_by_coords(grid_df, downstream_df, source_label):
    """
    Inner-merge a downstream file (with potentially ambiguous site_ids) onto
    the canonical 50-site grid by (latitude, longitude). Returns a DataFrame
    with the canonical site_ids from grid_df and all the data columns from
    downstream_df, sorted by site_id.

    Lat/lon are rounded to COORD_PRECISION decimals on both sides to avoid
    float comparison issues from CSV roundtripping.
    """
    grid = grid_df[['site_id', 'latitude', 'longitude']].copy()
    grid['_lat_key'] = grid['latitude'].round(COORD_PRECISION)
    grid['_lon_key'] = grid['longitude'].round(COORD_PRECISION)

    down = downstream_df.copy()
    if 'latitude' not in down.columns or 'longitude' not in down.columns:
        raise RuntimeError(
            f'{source_label} file is missing latitude/longitude columns; '
            f'cannot merge by coordinates.'
        )
    down['_lat_key'] = down['latitude'].round(COORD_PRECISION)
    down['_lon_key'] = down['longitude'].round(COORD_PRECISION)

    # Drop the (ambiguous) site_id from the downstream side; we will use the
    # canonical one from the grid. Also drop lat/lon to avoid duplicate
    # columns in the merge result.
    drop_cols = [c for c in ['site_id', 'latitude', 'longitude'] if c in down.columns]
    down_data = down.drop(columns=drop_cols)

    merged = grid.merge(down_data, on=['_lat_key', '_lon_key'], how='inner')
    merged = merged.drop(columns=['_lat_key', '_lon_key'])
    merged = merged.sort_values('site_id').reset_index(drop=True)

    return merged


def filter_nlcd(grid_50_df):
    """Merge NLCD onto the 50-site grid by (latitude, longitude)."""
    src = f'{NLCD_DIR}/site_nlcd_classes.csv'
    dst = f'{NLCD_DIR}/site_nlcd_classes_50sites.csv'

    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found.')

    nlcd = pd.read_csv(src)
    merged = merge_by_coords(grid_50_df, nlcd, 'NLCD')

    merged.to_csv(dst, index=False)
    print(f'  NLCD:  merged {len(nlcd)} -> {len(merged)} rows')
    print(f'         saved {dst}')

    if len(merged) != len(grid_50_df):
        print(f'  WARNING: expected {len(grid_50_df)} rows, got {len(merged)}')
        grid_coords = set(zip(
            grid_50_df['latitude'].round(COORD_PRECISION),
            grid_50_df['longitude'].round(COORD_PRECISION)
        ))
        nlcd_coords = set(zip(
            nlcd['latitude'].round(COORD_PRECISION),
            nlcd['longitude'].round(COORD_PRECISION)
        ))
        missing = grid_coords - nlcd_coords
        print(f'  Missing coordinates: {sorted(missing)}')

    return merged


def filter_osm_with_patch(grid_50_df):
    """
    Merge OSM onto the 50-site grid by (latitude, longitude) AND apply the
    OSM silent-failure patch (0.0 or NaN -> 10000 m).
    """
    src = f'{OSM_DIR}/site_osm_distances.csv'
    dst = f'{OSM_DIR}/site_osm_distances_50sites.csv'
    log_path = f'{OSM_DIR}/site_osm_imputation_log_50sites.csv'

    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found.')

    osm = pd.read_csv(src)
    merged = merge_by_coords(grid_50_df, osm, 'OSM')

    # Apply silent-failure patch
    log_rows = []
    for field in DISTANCE_FIELDS:
        if field not in merged.columns:
            print(f'  WARNING: column {field} not found in {src}')
            continue

        for idx, row in merged.iterrows():
            val = row[field]
            should_patch = False
            reason = None

            if pd.isna(val):
                should_patch = True
                reason = 'NaN'
            elif float(val) == 0.0:
                should_patch = True
                reason = 'exactly_zero'

            if should_patch:
                log_rows.append({
                    'site_id': row['site_id'],
                    'field': field,
                    'original_value': val,
                    'imputed_value': PATCH_VALUE,
                    'reason': reason,
                })
                merged.at[idx, field] = PATCH_VALUE

    # Recompute scores from patched distances
    merged['road_access_score'] = 1.0 - (merged['dist_road_m'].clip(0, 10000) / 10000)
    merged['water_source_score'] = 1.0 - (merged['dist_waterway_m'].clip(0, 10000) / 10000)

    merged.to_csv(dst, index=False)
    print(f'  OSM:   merged {len(osm)} -> {len(merged)} rows')
    print(f'         {len(log_rows)} field-level imputations applied')
    print(f'         saved {dst}')

    log_df = pd.DataFrame(
        log_rows,
        columns=['site_id', 'field', 'original_value', 'imputed_value', 'reason']
    )
    log_df.to_csv(log_path, index=False)
    print(f'         imputation log: {log_path}')

    if len(merged) != len(grid_50_df):
        print(f'  WARNING: expected {len(grid_50_df)} rows, got {len(merged)}')

    return merged, log_rows


def copy_nwis():
    """NWIS is county-scoped — just copy with the _50sites suffix."""
    for src_name, dst_name in [
        ('nwis_gw_sites.csv', 'nwis_gw_sites_50sites.csv'),
        ('nwis_gw_levels.csv', 'nwis_gw_levels_50sites.csv'),
    ]:
        src = f'{NWIS_DIR}/{src_name}'
        dst = f'{NWIS_DIR}/{dst_name}'
        if not os.path.exists(src):
            print(f'  WARNING: {src} not found, skipping')
            continue
        shutil.copy2(src, dst)
        print(f'  NWIS:  copied {src_name} -> {dst_name} (county-scoped)')


def copy_noaa():
    """NOAA is study-area-scoped — just copy with the _50sites suffix."""
    src = f'{NOAA_DIR}/noaa_daily_climate.csv'
    dst = f'{NOAA_DIR}/noaa_daily_climate_50sites.csv'
    if not os.path.exists(src):
        print(f'  WARNING: {src} not found, skipping')
        return
    shutil.copy2(src, dst)
    print(f'  NOAA:  copied noaa_daily_climate.csv -> noaa_daily_climate_50sites.csv')
    print(f'         (study-area-scoped; same Albany station data)')


def filter_ssurgo(mukeys_50_df):
    """
    Filter SSURGO files to the mukeys present in the 50-site grid.
    Soil properties and water table are mukey-keyed; texture is cokey-keyed
    and gets filtered through the soil_properties result.

    SSURGO is unaffected by the site_id collision bug because mukeys are soil
    unit identifiers, not coordinate identifiers. The same mukey can correctly
    appear in multiple grids without ambiguity.
    """
    needed_mukeys = set(mukeys_50_df['mukey'].dropna().astype(str))
    print(f'  SSURGO grid contains {len(needed_mukeys)} unique mukeys')

    # Soil properties (mukey-keyed)
    src = f'{SSURGO_DIR}/ssurgo_soil_properties.csv'
    dst = f'{SSURGO_DIR}/ssurgo_soil_properties_50sites.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found.')
    soil = pd.read_csv(src, dtype={'mukey': str})
    soil_50 = soil[soil['mukey'].isin(needed_mukeys)].reset_index(drop=True)
    soil_50.to_csv(dst, index=False)
    print(f'  SSURGO soil_properties:  filtered {len(soil)} -> {len(soil_50)} rows')

    # Texture (cokey-keyed; filter via soil_50 cokeys)
    src = f'{SSURGO_DIR}/ssurgo_texture.csv'
    dst = f'{SSURGO_DIR}/ssurgo_texture_50sites.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found.')
    tex = pd.read_csv(src)
    needed_cokeys = set(soil_50['cokey'].dropna().astype(str))
    tex_50 = tex[tex['cokey'].astype(str).isin(needed_cokeys)].reset_index(drop=True)
    tex_50.to_csv(dst, index=False)
    print(f'  SSURGO texture:          filtered {len(tex)} -> {len(tex_50)} rows')

    # Water table (mukey-keyed)
    src = f'{SSURGO_DIR}/ssurgo_water_table.csv'
    dst = f'{SSURGO_DIR}/ssurgo_water_table_50sites.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found.')
    wt = pd.read_csv(src, dtype={'mukey': str})
    wt_50 = wt[wt['mukey'].isin(needed_mukeys)].reset_index(drop=True)
    wt_50.to_csv(dst, index=False)
    print(f'  SSURGO water_table:      filtered {len(wt)} -> {len(wt_50)} rows')


if __name__ == '__main__':
    print('=' * 60)
    print('VQ-MAR — Build 50-Site Dataset from Existing Files')
    print('=' * 60)

    # Step 1: Load the canonical 50-site grid
    print()
    print('Loading 50-site grid (site_mukeys_50.csv)...')
    src_50 = f'{SSURGO_DIR}/site_mukeys_50.csv'
    if not os.path.exists(src_50):
        print(f'ERROR: {src_50} not found.')
        exit(1)
    mukeys_50 = pd.read_csv(src_50, dtype={'mukey': str})
    print(f'  {len(mukeys_50)} sites in 50-site grid')
    print(f'  {mukeys_50["mukey"].dropna().nunique()} unique mukeys')

    # Step 2: Save canonical 50-site grid file with _50sites suffix
    dst_50 = f'{SSURGO_DIR}/site_mukeys_50sites.csv'
    shutil.copy2(src_50, dst_50)
    print(f'  copied to {dst_50}')

    # Also save a coordinate-only candidate grid for downstream scripts
    candidate_grid = mukeys_50[['site_id', 'latitude', 'longitude']].copy()
    candidate_grid.to_csv(f'{SSURGO_DIR}/candidate_grid_50sites.csv', index=False)

    # Step 3: Filter NLCD via lat/lon merge
    print()
    print('Filtering NLCD by (latitude, longitude)...')
    filter_nlcd(mukeys_50)

    # Step 4: Filter OSM via lat/lon merge with silent-failure patch
    print()
    print('Filtering OSM by (latitude, longitude) with silent-failure patch...')
    osm_50, log_rows = filter_osm_with_patch(mukeys_50)

    # Step 5: NWIS — county-scoped, copy
    print()
    print('Copying NWIS (county-scoped, no filtering needed)...')
    copy_nwis()

    # Step 6: NOAA — study-area-scoped, copy
    print()
    print('Copying NOAA (study-area-scoped, no filtering needed)...')
    copy_noaa()

    # Step 7: Filter SSURGO files to 50-site mukeys
    print()
    print('Filtering SSURGO files to 50-site mukeys...')
    filter_ssurgo(mukeys_50)

    # Final summary
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'50-site dataset built from existing data files.')
    print(f'No API calls were made.')
    print()
    print(f'OSM imputations applied: {len(log_rows)}')
    if len(log_rows) > 0:
        print(f'  Review georgia/raw/osm/site_osm_imputation_log_50sites.csv')
        print(f'  for the audit trail.')
    print()
    print('Outputs:')
    print(f'  georgia/raw/ssurgo/site_mukeys_50sites.csv')
    print(f'  georgia/raw/ssurgo/candidate_grid_50sites.csv')
    print(f'  georgia/raw/ssurgo/ssurgo_soil_properties_50sites.csv')
    print(f'  georgia/raw/ssurgo/ssurgo_texture_50sites.csv')
    print(f'  georgia/raw/ssurgo/ssurgo_water_table_50sites.csv')
    print(f'  georgia/raw/nwis/nwis_gw_sites_50sites.csv')
    print(f'  georgia/raw/nwis/nwis_gw_levels_50sites.csv')
    print(f'  georgia/raw/nlcd/site_nlcd_classes_50sites.csv')
    print(f'  georgia/raw/noaa/noaa_daily_climate_50sites.csv')
    print(f'  georgia/raw/osm/site_osm_distances_50sites.csv')
    print(f'  georgia/raw/osm/site_osm_imputation_log_50sites.csv')
