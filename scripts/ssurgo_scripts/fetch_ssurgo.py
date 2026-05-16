"""
VQ-MAR — Fetch SSURGO soil data for Georgia pilot region.
 
IMPORTANT: Run fetch_ssurgo_spatial.py FIRST to generate the
site_mukeys_*.csv files. This script reads those files to know
which mukeys to query.
 
Queries SDA by mukey (not by county) to avoid sacatalog join issues.
Uses correct table/column names validated against live SDA endpoint:
  - component table: hydgrp, drainagecl, slope_r
  - chorizon table: ksat_r, awc_r, sandtotal_r, claytotal_r, etc.
  - chtexturegrp/chtexture tables: texcl (texture class)
  - muaggatt table: wtdepannmin (depth to water table)
"""
 
import requests
import pandas as pd
import os
import glob
import time
 
# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'ssurgo')
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
SDA_URL = 'https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest'
 
# Column names for each query (SDA returns NO headers)
SOIL_COLUMNS = [
    'cokey', 'mukey', 'comppct_r', 'compname',
    'hydgrp', 'drainagecl', 'slope_r',
    'chkey', 'hzdept_r', 'hzdepb_r',
    'ksat_r', 'awc_r',
    'sandtotal_r', 'claytotal_r',
    'dbthirdbar_r', 'om_r'
]
 
TEXTURE_COLUMNS = ['cokey', 'chkey', 'texdesc', 'texcl']
 
MUAGGATT_COLUMNS = ['mukey', 'wtdepannmin', 'brockdepmin', 'niccdcd']
 
 
def query_sda_raw(sql):
    """
    Query SDA and return the raw Table array.
    SDA returns NO header row -- every row is data.
    """
    payload = {'query': sql, 'format': 'JSON'}
    try:
        r = requests.post(SDA_URL, json=payload, timeout=60)
        if r.status_code != 200:
            print(f'  HTTP {r.status_code}: {r.text[:200]}')
            return []
        data = r.json()
        if 'Table' in data:
            return data['Table']
    except Exception as e:
        print(f'  Error: {e}')
    return []
 
 
def collect_mukeys():
    """Read unique mukeys from all site_mukeys_*.csv files."""
    pattern = f'{OUTPUT_DIR}/site_mukeys_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'ERROR: No site_mukeys files found in {OUTPUT_DIR}/')
        print('Run fetch_ssurgo_spatial.py first.')
        return []
 
    all_mukeys = set()
    for f in files:
        df = pd.read_csv(f)
        mukeys = df['mukey'].dropna().astype(str).unique()
        all_mukeys.update(mukeys)
        print(f'  {os.path.basename(f)}: {len(mukeys)} unique mukeys')
 
    mukeys = sorted(all_mukeys)
    print(f'  Total unique mukeys across all grids: {len(mukeys)}')
    return mukeys
 
 
def fetch_soil_properties(mukeys):
    """Fetch component + chorizon data for a list of mukeys."""
    all_rows = []
    batch_size = 20
 
    for i in range(0, len(mukeys), batch_size):
        batch = mukeys[i:i + batch_size]
        in_clause = ','.join(f"'{m}'" for m in batch)
 
        sql = (
            f"SELECT c.cokey, c.mukey, c.comppct_r, c.compname, "
            f"c.hydgrp, c.drainagecl, c.slope_r, "
            f"ch.chkey, ch.hzdept_r, ch.hzdepb_r, "
            f"ch.ksat_r, ch.awc_r, "
            f"ch.sandtotal_r, ch.claytotal_r, "
            f"ch.dbthirdbar_r, ch.om_r "
            f"FROM component c "
            f"LEFT JOIN chorizon ch ON ch.cokey = c.cokey "
            f"WHERE c.mukey IN ({in_clause}) "
            f"AND c.comppct_r >= 15 "
            f"ORDER BY c.mukey, c.comppct_r DESC, ch.hzdept_r"
        )
 
        rows = query_sda_raw(sql)
        all_rows.extend(rows)
        print(f'  Batch {i // batch_size + 1}: {len(rows)} records')
        time.sleep(0.3)
 
    if all_rows:
        return pd.DataFrame(all_rows, columns=SOIL_COLUMNS)
    return pd.DataFrame(columns=SOIL_COLUMNS)
 
 
def fetch_texture_classes(mukeys):
    """Fetch texture class from chtexturegrp/chtexture tables."""
    all_rows = []
    batch_size = 20
 
    for i in range(0, len(mukeys), batch_size):
        batch = mukeys[i:i + batch_size]
        in_clause = ','.join(f"'{m}'" for m in batch)
 
        sql = (
            f"SELECT ch.cokey, ch.chkey, "
            f"chtg.texdesc, cht.texcl "
            f"FROM component c "
            f"LEFT JOIN chorizon ch ON ch.cokey = c.cokey "
            f"LEFT JOIN chtexturegrp chtg ON chtg.chkey = ch.chkey "
            f"LEFT JOIN chtexture cht ON cht.chtgkey = chtg.chtgkey "
            f"WHERE c.mukey IN ({in_clause}) "
            f"AND c.comppct_r >= 15 "
            f"AND chtg.rvindicator = 'Yes'"
        )
 
        rows = query_sda_raw(sql)
        all_rows.extend(rows)
        time.sleep(0.3)
 
    if all_rows:
        return pd.DataFrame(all_rows, columns=TEXTURE_COLUMNS)
    return pd.DataFrame(columns=TEXTURE_COLUMNS)
 
 
def fetch_water_table(mukeys):
    """Fetch water table depth from muaggatt table."""
    all_rows = []
    batch_size = 50
 
    for i in range(0, len(mukeys), batch_size):
        batch = mukeys[i:i + batch_size]
        in_clause = ','.join(f"'{m}'" for m in batch)
 
        sql = (
            f"SELECT mukey, wtdepannmin, brockdepmin, niccdcd "
            f"FROM muaggatt "
            f"WHERE mukey IN ({in_clause})"
        )
 
        rows = query_sda_raw(sql)
        all_rows.extend(rows)
        time.sleep(0.3)
 
    if all_rows:
        return pd.DataFrame(all_rows, columns=MUAGGATT_COLUMNS)
    return pd.DataFrame(columns=MUAGGATT_COLUMNS)
 
 
if __name__ == '__main__':
    # Step 1: Collect mukeys from spatial query results
    print('Collecting mukeys from site grids...')
    mukeys = collect_mukeys()
    if not mukeys:
        exit(1)
 
    # Step 2: Fetch soil properties
    print()
    print(f'Fetching soil properties for {len(mukeys)} mukeys...')
    soil_df = fetch_soil_properties(mukeys)
    soil_df.to_csv(f'{OUTPUT_DIR}/ssurgo_soil_properties.csv', index=False)
    print(f'  Total: {len(soil_df)} soil records')
 
    # Step 3: Fetch texture classes
    print()
    print(f'Fetching texture classes...')
    texture_df = fetch_texture_classes(mukeys)
    texture_df.to_csv(f'{OUTPUT_DIR}/ssurgo_texture.csv', index=False)
    print(f'  Total: {len(texture_df)} texture records')
 
    # Step 4: Fetch water table depth
    print()
    print(f'Fetching water table depth...')
    wt_df = fetch_water_table(mukeys)
    wt_df.to_csv(f'{OUTPUT_DIR}/ssurgo_water_table.csv', index=False)
    print(f'  Total: {len(wt_df)} water table records')
 
    # Summary
    print()
    print(f'All data saved to {OUTPUT_DIR}/')
    print(f'  ssurgo_soil_properties.csv:  {len(soil_df)} rows')
    print(f'  ssurgo_texture.csv:          {len(texture_df)} rows')
    print(f'  ssurgo_water_table.csv:      {len(wt_df)} rows')
