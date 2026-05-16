"""
VQ-MAR — Assign SSURGO map units to candidate site locations.
 
For each candidate site grid point, queries SDA to find which SSURGO
map unit (mukey) is at that location. The mukey is then used to join
to the tabular soil properties from fetch_ssurgo.py.
 
Uses the SDA stored procedure SDA_Get_Mukey_from_intersection_with_WktWgs84
which is reliable where the WFS and spatial SQL endpoints are not.
"""
 
import requests
import pandas as pd
import numpy as np
import os
import time
 
# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
 
# Study area bounding box
BBOX = {
    'north': 32.50, 'south': 31.60,
    'east': -83.60, 'west': -84.60
}
 
 
def query_sda(sql):
    """Execute SQL query against SDA POST endpoint."""
    payload = {"query": sql, "format": "JSON"}
    response = requests.post(SDA_URL, json=payload)
    if response.status_code != 200:
        print(f'  HTTP {response.status_code}: {response.text[:300]}')
        return None
    data = response.json()
    if "Table" in data and len(data["Table"]) > 1:
        columns = data["Table"][0]
        rows = data["Table"][1:]
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame()
 
 
def generate_candidate_grid(n_sites):
    """
    Generate a grid of candidate site locations within the study area.
    Returns a DataFrame with site_id, latitude, longitude.
    """
    # Calculate grid dimensions to get approximately n_sites points
    aspect = (BBOX['east'] - BBOX['west']) / (BBOX['north'] - BBOX['south'])
    n_cols = int(np.ceil(np.sqrt(n_sites * aspect)))
    n_rows = int(np.ceil(n_sites / n_cols))
 
    lats = np.linspace(BBOX['south'] + 0.05, BBOX['north'] - 0.05, n_rows)
    lons = np.linspace(BBOX['west'] + 0.05, BBOX['east'] - 0.05, n_cols)
 
    sites = []
    site_id = 1
    for lat in lats:
        for lon in lons:
            if site_id > n_sites:
                break
            sites.append({
                'site_id': f'GA_{site_id:03d}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6)
            })
            site_id += 1
 
    return pd.DataFrame(sites)
 
 
def get_mukey_at_point(lat, lon):
    """
    Query SDA for the SSURGO map unit key at a given lat/lon.
    Uses the SDA stored procedure for point-in-polygon lookup.
 
    The stored procedure returns {'Table':[['mukey_value']]}
    with NO header row, so we parse it directly rather than
    using query_sda() which expects a header row.
    """
    sql = f"""
    SELECT *
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
        'POINT({lon} {lat})'
    )
    """
    payload = {"query": sql, "format": "JSON"}
    try:
        response = requests.post(SDA_URL, json=payload, timeout=30)
        if response.status_code != 200:
            return None
        data = response.json()
        if 'Table' in data and len(data['Table']) > 0:
            row = data['Table'][0]
            if row and row[0]:
                return row[0]  # Returns the mukey string
    except Exception:
        pass
    return None
 
 
def assign_mukeys_to_grid(grid_df):
    """
    For each site in the grid, query SDA for its mukey.
    Includes rate limiting to avoid overloading SDA.
    """
    results = []
    total = len(grid_df)
 
    for idx, row in grid_df.iterrows():
        site_id = row['site_id']
        lat = row['latitude']
        lon = row['longitude']
 
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'  Querying site {idx + 1}/{total}...')
 
        mukey = get_mukey_at_point(lat, lon)
 
        results.append({
            'site_id': site_id,
            'latitude': lat,
            'longitude': lon,
            'mukey': mukey
        })
 
        time.sleep(0.2)  # Rate limit: ~5 requests/second
 
    return pd.DataFrame(results)
 
 
if __name__ == '__main__':
    # Generate candidate site grids for all problem sizes
    for n_sites in [20, 50]:
        print()
        print('=' * 60)
        print(f'Generating {n_sites}-site candidate grid...')
        print('=' * 60)
 
        grid = generate_candidate_grid(n_sites)
        print(f'  Created {len(grid)} candidate sites')
 
        # Save the grid coordinates
        grid_path = f'{OUTPUT_DIR}/candidate_grid_{n_sites}.csv'
        grid.to_csv(grid_path, index=False)
        print(f'  Grid saved to {grid_path}')
 
        # Query mukeys for each site
        print(f'  Querying SDA for mukeys ({len(grid)} points)...')
        site_mukeys = assign_mukeys_to_grid(grid)
 
        # Report results
        n_found = site_mukeys['mukey'].notna().sum()
        n_missing = site_mukeys['mukey'].isna().sum()
        print(f'  Results: {n_found} sites with mukeys, {n_missing} without')
 
        # Save
        out_path = f'{OUTPUT_DIR}/site_mukeys_{n_sites}.csv'
        site_mukeys.to_csv(out_path, index=False)
        print(f'  Saved to {out_path}')
 
        if n_missing > 0:
            print(f'  Sites without mukeys are likely in water or developed areas.')
            print(f'  These will be hard-excluded in the QUBO.')
 
    # Show how to join with tabular data
    print()
    print('=' * 60)
    print('Next step: join site_mukeys with ssurgo_soil_properties.csv')
    print('using the mukey column to assign soil properties to each site.')
    print('=' * 60)
    print()
    print(f'All data saved to {OUTPUT_DIR}/')
