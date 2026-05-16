"""
VQ-MAR — Fetch USGS NWIS groundwater data for Georgia pilot.
No registration or API key required.
"""
 
import dataretrieval.nwis as nwis
import pandas as pd
import os
 
# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'nwis')
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Study area county codes (5-digit FIPS: state 13 = Georgia + county)
COUNTIES = ['13261', '13177', '13273', '13307', '13249', '13193',
            '13093', '13081', '13321', '13095']
 
 
def fetch_site_info():
    """Fetch groundwater well metadata for all study area counties."""
    all_sites = []
    for county in COUNTIES:
        try:
            sites_df, _ = nwis.get_info(
                countyCd=county,
                siteType='GW',
                siteStatus='all'
            )
            sites_df['county_cd'] = county
            all_sites.append(sites_df)
            print(f'  County {county}: {len(sites_df)} wells')
        except Exception as e:
            print(f'  County {county}: {e}')
    if all_sites:
        return pd.concat(all_sites, ignore_index=True)
    return pd.DataFrame()
 
 
def fetch_groundwater_levels(site_numbers):
    """Fetch daily groundwater level time series in batches."""
    batch_size = 100
    all_data = []
 
    for i in range(0, len(site_numbers), batch_size):
        batch = site_numbers[i:i + batch_size]
        print(f'  Batch {i // batch_size + 1}: {len(batch)} sites...')
 
        # Parameter 72019: depth to water level, ft below surface
        try:
            gw_df, _ = nwis.get_dv(
                sites=batch,
                parameterCd='72019',
                start='1990-01-01',
                end='2026-03-01'
            )
            all_data.append(gw_df)
        except Exception as e:
            print(f'    Depth data: {e}')
 
        # Parameter 72020: water level elevation, ft above NGVD29
        try:
            elev_df, _ = nwis.get_dv(
                sites=batch,
                parameterCd='72020',
                start='1990-01-01',
                end='2026-03-01'
            )
            all_data.append(elev_df)
        except Exception as e:
            print(f'    Elevation data: {e}')
 
    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()
 
 
if __name__ == '__main__':
    print('Fetching groundwater well metadata...')
    sites = fetch_site_info()
    sites.to_csv(f'{OUTPUT_DIR}/nwis_gw_sites.csv', index=False)
    print(f'Total wells: {len(sites)}')
 
    if len(sites) == 0:
        print('ERROR: No wells found. Check county codes.')
        exit(1)
 
    site_numbers = sites['site_no'].unique().tolist()
    print()
    print(f'Fetching groundwater levels for {len(site_numbers)} sites...')
    gw_data = fetch_groundwater_levels(site_numbers)
 
    if len(gw_data) > 0:
        gw_data.to_csv(f'{OUTPUT_DIR}/nwis_gw_levels.csv')
        print(f'Saved {len(gw_data)} water level records')
    else:
        print('No groundwater level data retrieved')
 
    print()
    print(f'All data saved to {OUTPUT_DIR}/')
