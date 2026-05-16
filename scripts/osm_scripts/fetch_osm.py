"""
VQ-MAR — Fetch OpenStreetMap infrastructure data for Georgia sites.

Queries the Overpass API directly (not via osmnx bbox) for infrastructure
features near each candidate site. Uses a per-site radius query to stay
within Overpass API limits.

No registration required.
"""

import requests
import pandas as pd
import numpy as np
import os
import glob
import time
import math

# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

SSURGO_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'osm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

# Search radius around each site (meters)
SEARCH_RADIUS = 10000  # 10 km


def load_sites():
    """Load unique candidate site locations."""
    pattern = f'{SSURGO_DIR}/site_mukeys_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'ERROR: No site_mukeys files found in {SSURGO_DIR}/')
        return None

    all_dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        all_dfs.append(df)
        print(f'  {os.path.basename(f)}: {len(df)} sites')

    combined = pd.concat(all_dfs, ignore_index=True)
    unique = combined.drop_duplicates(
        subset=['latitude', 'longitude']
    )[['site_id', 'latitude', 'longitude']].reset_index(drop=True)
    print(f'  Total unique locations: {len(unique)}')
    return unique


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def query_overpass_batch(south, west, north, east, query_type):
    """
    Query Overpass API for features in a bounding box.
    Returns list of (lat, lon, type, name) tuples.
    """
    if query_type == 'roads':
        filters = '["highway"~"primary|secondary|tertiary|trunk|motorway"]'
        query = f"""
        [out:json][timeout:60];
        (
          way{filters}({south},{west},{north},{east});
        );
        out center;
        """
    elif query_type == 'waterways':
        query = f"""
        [out:json][timeout:60];
        (
          way["waterway"~"river|stream|canal"]({south},{west},{north},{east});
        );
        out center;
        """
    elif query_type == 'water_infra':
        query = f"""
        [out:json][timeout:60];
        (
          node["man_made"~"water_well|water_tower"]({south},{west},{north},{east});
          node["amenity"="drinking_water"]({south},{west},{north},{east});
        );
        out;
        """
    else:
        return []

    try:
        r = requests.post(OVERPASS_URL, data={'data': query}, timeout=90)
        if r.status_code == 200:
            data = r.json()
            results = []
            for elem in data.get('elements', []):
                lat = elem.get('lat') or elem.get('center', {}).get('lat')
                lon = elem.get('lon') or elem.get('center', {}).get('lon')
                if lat and lon:
                    tags = elem.get('tags', {})
                    ftype = (tags.get('highway') or tags.get('waterway') or
                             tags.get('man_made') or tags.get('amenity') or 'unknown')
                    fname = tags.get('name', '')
                    results.append((lat, lon, ftype, fname))
            return results
        elif r.status_code == 429:
            print('    Rate limited, waiting 30s...')
            time.sleep(30)
            return query_overpass_batch(south, west, north, east, query_type)
        else:
            print(f'    Overpass error: {r.status_code}')
            return []
    except requests.exceptions.Timeout:
        print(f'    Timeout for {query_type}')
        return []
    except Exception as e:
        print(f'    Error: {e}')
        return []


def fetch_all_features():
    """
    Fetch features using a grid of overlapping tiles to stay within
    Overpass limits. Tiles are ~0.3 degrees (~33 km) on a side.
    """
    # Study area
    south, north = 31.60, 32.50
    west, east = -84.60, -83.60

    tile_size = 0.3  # degrees
    lat_tiles = np.arange(south, north, tile_size)
    lon_tiles = np.arange(west, east, tile_size)

    all_roads = []
    all_waterways = []
    all_water_infra = []

    total_tiles = len(lat_tiles) * len(lon_tiles)
    tile_num = 0

    for lat_start in lat_tiles:
        for lon_start in lon_tiles:
            tile_num += 1
            s = lat_start
            n = min(lat_start + tile_size, north)
            w = lon_start
            e = min(lon_start + tile_size, east)

            print(f'  Tile {tile_num}/{total_tiles}: '
                  f'[{s:.2f}-{n:.2f}N, {w:.2f}-{e:.2f}W]...', end=' ')

            # Roads
            roads = query_overpass_batch(s, w, n, e, 'roads')
            all_roads.extend(roads)
            time.sleep(1)

            # Waterways
            waterways = query_overpass_batch(s, w, n, e, 'waterways')
            all_waterways.extend(waterways)
            time.sleep(1)

            # Water infrastructure
            water_infra = query_overpass_batch(s, w, n, e, 'water_infra')
            all_water_infra.extend(water_infra)
            time.sleep(1)

            print(f'roads={len(roads)}, waterways={len(waterways)}, '
                  f'water={len(water_infra)}')

    # Deduplicate by rounding coordinates
    def dedup(features):
        seen = set()
        unique = []
        for lat, lon, ftype, fname in features:
            key = (round(lat, 5), round(lon, 5), ftype)
            if key not in seen:
                seen.add(key)
                unique.append((lat, lon, ftype, fname))
        return unique

    all_roads = dedup(all_roads)
    all_waterways = dedup(all_waterways)
    all_water_infra = dedup(all_water_infra)

    print(f'\nAfter deduplication:')
    print(f'  Roads: {len(all_roads)} segments')
    print(f'  Waterways: {len(all_waterways)} segments')
    print(f'  Water infrastructure: {len(all_water_infra)} points')

    return all_roads, all_waterways, all_water_infra


def compute_distances(sites, roads, waterways, water_infra):
    """Compute distance from each site to nearest feature of each type."""
    results = sites.copy()

    def nearest_distance(lat, lon, features):
        if not features:
            return np.nan
        return min(haversine_m(lat, lon, f[0], f[1]) for f in features)

    print(f'Computing distances for {len(sites)} sites...')

    dist_road = []
    dist_water = []
    dist_infra = []

    for idx, row in sites.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f'  Site {idx + 1}/{len(sites)}...')

        lat, lon = row['latitude'], row['longitude']
        dist_road.append(nearest_distance(lat, lon, roads))
        dist_water.append(nearest_distance(lat, lon, waterways))
        dist_infra.append(nearest_distance(lat, lon, water_infra))

    results['dist_road_m'] = dist_road
    results['dist_waterway_m'] = dist_water
    results['dist_water_infra_m'] = dist_infra

    # Composite access score: closer to road = better access = lower cost
    results['road_access_score'] = (
        1.0 - (results['dist_road_m'].clip(0, 10000) / 10000)
    )

    # Water source proximity score: closer to water = easier to supply
    results['water_source_score'] = (
        1.0 - (results['dist_waterway_m'].clip(0, 10000) / 10000)
    )

    return results


def save_features_csv(features, ftype, output_dir):
    """Save feature list as CSV for reference."""
    if features:
        df = pd.DataFrame(features, columns=['latitude', 'longitude', 'type', 'name'])
        outpath = f'{output_dir}/osm_{ftype}.csv'
        df.to_csv(outpath, index=False)
        print(f'  Saved {len(df)} {ftype} features to {outpath}')


if __name__ == '__main__':
    print('Loading candidate site grids...')
    sites = load_sites()
    if sites is None:
        exit(1)

    print()
    print('Fetching OSM features in tiled queries...')
    roads, waterways, water_infra = fetch_all_features()

    # Save raw features for reference
    print()
    save_features_csv(roads, 'roads', OUTPUT_DIR)
    save_features_csv(waterways, 'waterways', OUTPUT_DIR)
    save_features_csv(water_infra, 'water_infrastructure', OUTPUT_DIR)

    # Compute distances
    print()
    results = compute_distances(sites, roads, waterways, water_infra)

    # Save site distances
    outpath = f'{OUTPUT_DIR}/site_osm_distances.csv'
    results.to_csv(outpath, index=False)

    # Report
    print()
    print('=' * 60)
    print('Infrastructure distance summary:')
    print('=' * 60)

    for col, label in [('dist_road_m', 'Nearest major road'),
                       ('dist_waterway_m', 'Nearest waterway'),
                       ('dist_water_infra_m', 'Nearest water infrastructure')]:
        vals = results[col].dropna()
        if len(vals) > 0:
            print(f'  {label}:')
            print(f'    Mean: {vals.mean():.0f} m')
            print(f'    Median: {vals.median():.0f} m')
            print(f'    Range: [{vals.min():.0f}, {vals.max():.0f}] m')
            if col == 'dist_road_m':
                print(f'    Within 1 km: {(vals < 1000).sum()} sites')
                print(f'    Within 5 km: {(vals < 5000).sum()} sites')

    rs = results['road_access_score']
    print(f'  Road access score (0=worst, 1=best):')
    print(f'    Mean: {rs.mean():.2f}')

    print()
    print(f'Saved to {outpath}')