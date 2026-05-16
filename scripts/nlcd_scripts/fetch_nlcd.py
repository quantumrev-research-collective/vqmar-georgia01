"""
VQ-MAR — Fetch NLCD land cover for Georgia candidate sites.

Queries the MRLC WMS (Web Map Service) to get the NLCD land cover
class and impervious surface percentage at each candidate site point.
Uses NLCD 2021 (the most recent year available on the WMS).

No download, no registration, no AWS credentials needed.
"""

import requests
import pandas as pd
import numpy as np
import os
import glob
import time
import struct

# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

SSURGO_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'nlcd')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MRLC WMS endpoint
WMS_URL = 'https://www.mrlc.gov/geoserver/mrlc_display/wms'

# WMS layer names (from GetCapabilities)
LAND_COVER_LAYER = 'NLCD_2021_Land_Cover_L48'
IMPERVIOUS_LAYER = 'NLCD_2021_Impervious_L48'

# NLCD class definitions
NLCD_CLASSES = {
    11: 'Open Water', 12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space', 22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity', 24: 'Developed, High Intensity',
    31: 'Barren Land',
    41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest',
    52: 'Shrub/Scrub', 71: 'Grassland/Herbaceous',
    81: 'Pasture/Hay', 82: 'Cultivated Crops',
    90: 'Woody Wetlands', 95: 'Emergent Herbaceous Wetlands',
}

# MAR suitability scores per NLCD class
SUITABILITY_SCORES = {
    11: 0.0, 12: 0.0,
    21: 0.2, 22: 0.0, 23: 0.0, 24: 0.0,
    31: 0.8,
    41: 0.6, 42: 0.6, 43: 0.6,
    52: 0.8, 71: 0.9,
    81: 0.7, 82: 0.5,
    90: 0.1, 95: 0.1,
}


def query_wms_getfeatureinfo(lat, lon, layer):
    """
    Query WMS GetFeatureInfo for a pixel value at a lat/lon point.
    Returns the pixel value (int) or None on failure.
    """
    delta = 0.001
    bbox = f'{lon - delta},{lat - delta},{lon + delta},{lat + delta}'

    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.1.1',
        'REQUEST': 'GetFeatureInfo',
        'LAYERS': layer,
        'QUERY_LAYERS': layer,
        'INFO_FORMAT': 'application/json',
        'SRS': 'EPSG:4326',
        'BBOX': bbox,
        'WIDTH': '3',
        'HEIGHT': '3',
        'X': '1',
        'Y': '1',
    }

    try:
        r = requests.get(WMS_URL, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if 'features' in data and len(data['features']) > 0:
                props = data['features'][0].get('properties', {})
                for key in ['GRAY_INDEX', 'PALETTE_INDEX', 'value', 'Value']:
                    if key in props:
                        return int(float(props[key]))
                # Try first numeric value in properties
                for v in props.values():
                    try:
                        val = int(float(v))
                        if 0 <= val <= 100:
                            return val
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return None


def query_wms_getmap(lat, lon, layer):
    """
    Fallback: Use WMS GetMap to get a single-pixel GeoTIFF and read it.
    Returns the pixel value (int) or None on failure.
    """
    delta = 0.0005
    bbox = f'{lon - delta},{lat - delta},{lon + delta},{lat + delta}'

    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.1.1',
        'REQUEST': 'GetMap',
        'LAYERS': layer,
        'SRS': 'EPSG:4326',
        'BBOX': bbox,
        'WIDTH': '1',
        'HEIGHT': '1',
        'FORMAT': 'image/tiff',
    }

    try:
        r = requests.get(WMS_URL, params=params, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            try:
                import rasterio
                import io
                with rasterio.open(io.BytesIO(r.content)) as src:
                    val = src.read(1)[0, 0]
                    if val > 0:
                        return int(val)
            except Exception:
                pass
    except Exception:
        pass
    return None


def get_pixel_value(lat, lon, layer):
    """Try GetFeatureInfo first, then GetMap as fallback."""
    val = query_wms_getfeatureinfo(lat, lon, layer)
    if val is None:
        val = query_wms_getmap(lat, lon, layer)
    return val


def collect_sites():
    """Read candidate site coordinates from the SSURGO grid files."""
    pattern = f'{SSURGO_DIR}/site_mukeys_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'ERROR: No site_mukeys files found in {SSURGO_DIR}/')
        print('Run fetch_ssurgo_spatial.py first.')
        return None

    all_points = []
    for f in sorted(files):
        df = pd.read_csv(f)
        all_points.append(df)
        print(f'  {os.path.basename(f)}: {len(df)} sites')

    combined = pd.concat(all_points, ignore_index=True)
    unique = combined.drop_duplicates(
        subset=['latitude', 'longitude']
    )[['site_id', 'latitude', 'longitude']].reset_index(drop=True)
    print(f'  Total unique locations: {len(unique)}')
    return unique


if __name__ == '__main__':
    print('Loading candidate site grids...')
    sites = collect_sites()
    if sites is None:
        exit(1)

    total = len(sites)

    # === Query land cover ===
    print()
    print(f'Querying NLCD 2021 Land Cover at {total} points...')
    print(f'  Layer: {LAND_COVER_LAYER}')

    lc_values = []
    for idx, row in sites.iterrows():
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'  Point {idx + 1}/{total}...')

        val = get_pixel_value(row['latitude'], row['longitude'], LAND_COVER_LAYER)
        lc_values.append(val)
        time.sleep(0.2)

    sites['nlcd_class'] = lc_values
    sites['nlcd_class_name'] = sites['nlcd_class'].map(
        lambda x: NLCD_CLASSES.get(x, 'Unknown') if pd.notna(x) else 'No data'
    )
    sites['mar_suitability_score'] = sites['nlcd_class'].map(
        lambda x: SUITABILITY_SCORES.get(x, 0.5) if pd.notna(x) else None
    )

    n_found = sites['nlcd_class'].notna().sum()
    print(f'  Land cover results: {n_found}/{total} sites')

    # === Query impervious surface ===
    print()
    print(f'Querying NLCD 2021 Impervious Surface at {total} points...')
    print(f'  Layer: {IMPERVIOUS_LAYER}')

    imp_values = []
    for idx, row in sites.iterrows():
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'  Point {idx + 1}/{total}...')

        val = get_pixel_value(row['latitude'], row['longitude'], IMPERVIOUS_LAYER)
        imp_values.append(val)
        time.sleep(0.2)

    sites['impervious_pct'] = imp_values

    n_imp = sites['impervious_pct'].notna().sum()
    print(f'  Impervious results: {n_imp}/{total} sites')

    # === Save results ===
    outpath = f'{OUTPUT_DIR}/site_nlcd_classes.csv'
    sites.to_csv(outpath, index=False)

    # === Report ===
    print()
    print('='*60)
    print('Land cover distribution across candidate sites:')
    print('='*60)
    for cls, group in sites.groupby('nlcd_class'):
        if pd.notna(cls):
            cls = int(cls)
            name = NLCD_CLASSES.get(cls, 'Unknown')
            score = SUITABILITY_SCORES.get(cls, 0.5)
            print(f'  {cls:3d} {name:40s} {len(group):3d} sites  score={score:.1f}')

    if n_imp > 0:
        imp = sites['impervious_pct'].dropna()
        print()
        print('Impervious surface statistics:')
        print(f'  Mean: {imp.mean():.1f}%')
        print(f'  Sites with >0%: {(imp > 0).sum()}')
        print(f'  Sites with >30%: {(imp > 30).sum()}')

    print()
    print(f'Saved to {outpath}')
    print(f'Columns: site_id, latitude, longitude, nlcd_class,')
    print(f'         nlcd_class_name, mar_suitability_score, impervious_pct')
    