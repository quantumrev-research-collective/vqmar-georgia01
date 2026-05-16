"""
VQ-MAR — Fetch NOAA GHCN-Daily climate data for Georgia study area.

Uses the NCEI Access Data Service v1 API to retrieve precipitation and
temperature from the Albany Regional Airport station (USW00013869), the
only GHCN-Daily station in the study area with continuous coverage.

No authentication required — NCEI deprecated tokens for this endpoint.
Docs: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
"""

import io
import os
import sys

import pandas as pd
import requests

# --- Path resolution ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
BASE_DIR = os.environ.get('VQMAR_BASE', _REPO_ROOT)

OUTPUT_DIR = os.path.join(BASE_DIR, 'georgia', 'raw', 'noaa')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Station — drop the old CDO "GHCND:" prefix; the new API uses bare IDs
STATION = "USW00013869"                              # Albany SW Georgia Regional Airport
STATION_NAME = "Albany SW Georgia Regional Airport"
STATION_LAT = 31.5353
STATION_LON = -84.1944

# Data parameters
DATATYPES = ["PRCP", "TMAX", "TMIN"]
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"


def fetch_daily_csv():
    """Fetch the full date range in a single CSV request and return a DataFrame."""
    params = {
        "dataset": "daily-summaries",
        "stations": STATION,
        "dataTypes": ",".join(DATATYPES),
        "startDate": START_DATE,
        "endDate": END_DATE,
        "units": "metric",              # PRCP -> mm, TMAX/TMIN -> °C
        "format": "csv",
        "includeStationName": "true",
        "includeStationLocation": "true",
    }

    print(f"Requesting {BASE_URL}")
    print(f"  station={STATION}  range={START_DATE}..{END_DATE}  dataTypes={DATATYPES}")
    r = requests.get(BASE_URL, params=params, timeout=300)

    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)

    if not r.text.strip():
        print("ERROR: empty response body")
        sys.exit(1)

    df = pd.read_csv(io.StringIO(r.text))
    print(f"  Received {len(df)} rows, columns: {list(df.columns)}")
    return df


if __name__ == '__main__':
    print("Fetching NOAA daily climate data...")
    raw = fetch_daily_csv()

    # Normalize column names to the schema transform_noaa.py expects.
    # The new API returns uppercase DATE/PRCP/TMAX/TMIN; the downstream
    # transform expects date/PRCP_mm/TMAX_C/TMIN_C.
    rename_map = {
        'DATE': 'date',
        'PRCP': 'PRCP_mm',
        'TMAX': 'TMAX_C',
        'TMIN': 'TMIN_C',
        'STATION': 'station',
        'NAME': 'station_name',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
    }
    pivot = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    # If station identity columns are missing (API omits them when a single
    # station is requested), fill from constants for downstream traceability.
    if 'station' not in pivot.columns:
        pivot['station'] = STATION
    if 'station_name' not in pivot.columns:
        pivot['station_name'] = STATION_NAME
    if 'latitude' not in pivot.columns:
        pivot['latitude'] = STATION_LAT
    if 'longitude' not in pivot.columns:
        pivot['longitude'] = STATION_LON

    pivot['date'] = pd.to_datetime(pivot['date'])
    pivot = pivot.sort_values('date').reset_index(drop=True)

    outpath = f'{OUTPUT_DIR}/noaa_daily_climate.csv'
    pivot.to_csv(outpath, index=False)

    # Report
    print()
    print(f"Date range: {pivot['date'].min().date()} to {pivot['date'].max().date()}")
    print(f"Total days: {len(pivot)}")

    if 'PRCP_mm' in pivot.columns:
        prcp = pd.to_numeric(pivot['PRCP_mm'], errors='coerce').dropna()
        years = (pivot['date'].max() - pivot['date'].min()).days / 365.25
        print(f"\nPrecipitation (PRCP):")
        print(f"  Records: {len(prcp)}")
        print(f"  Annual mean: {prcp.sum() / years:.0f} mm/yr")
        print(f"  Max daily: {prcp.max():.1f} mm")
        print(f"  Days with rain (>0.1 mm): {(prcp > 0.1).sum()}")

    if 'TMAX_C' in pivot.columns:
        tmax = pd.to_numeric(pivot['TMAX_C'], errors='coerce').dropna()
        print(f"\nMax Temperature (TMAX):")
        print(f"  Records: {len(tmax)}")
        print(f"  Mean: {tmax.mean():.1f} C")
        print(f"  Range: [{tmax.min():.1f}, {tmax.max():.1f}] C")

    if 'TMIN_C' in pivot.columns:
        tmin = pd.to_numeric(pivot['TMIN_C'], errors='coerce').dropna()
        print(f"\nMin Temperature (TMIN):")
        print(f"  Records: {len(tmin)}")
        print(f"  Mean: {tmin.mean():.1f} C")
        print(f"  Range: [{tmin.min():.1f}, {tmin.max():.1f}] C")

    print(f"\nSaved to {outpath}")