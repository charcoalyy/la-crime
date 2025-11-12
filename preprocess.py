import numpy as np
import pandas as pd
from dataclasses import dataclass

# pd.set_option('display.max_columns', None)

'''
AI DISCLAIMER NOTE: 
14 prompts used, where use cases are noted in the code below. Carbon usage for this file:

14*4.32g = 60.48g CO2

'''

# ====== constants ======
FILE_PATH = "Crime_Data_2010_2017.csv"
LA_LAT_MIN, LA_LAT_MAX = 33.0, 35.0
LA_LON_MIN, LA_LON_MAX = -119.5, -117.0

def debug(df, msg=""):
    print(f"\nDEBUG >> {msg}")
    print(df.head(10))

''' Automated manual work of creating data class, used ChatGPT assistance '''
@dataclass(frozen=True)
class raw_c:
    date: str = 'Date Occurred'
    time: str = 'Time Occurred'
    area: str = 'Area Name'
    crime_desc: str = 'Crime Code Description'
    location: str = 'Location'

''' Automated manual work of creating data class, used ChatGPT assistance '''
@dataclass(frozen=True)
class feat_c:
    datetime: str = 'datetime'
    lat: str = 'lat'
    lon: str = 'lon'

    grid_row: str = 'grid_row'
    grid_col: str = 'grid_col'
    grid_id: str = 'grid_id'

    week_number: str = 'week_number'
    week_year: str = 'week_year'
    week_start: str = 'week_start'
    week_end: str = 'week_end'
    week_id: str = 'week_id'

# ====== cleaning ======
def clean(df):
    """trim spaces from col names, keep only relevant cols, drop incomplete rows"""

    df.columns = df.columns.str.strip().str.title()
    df = df[[raw_c.date, raw_c.time, raw_c.area, raw_c.crime_desc, raw_c.location]]
    df = df.dropna(subset=[raw_c.date, raw_c.time, raw_c.location])
    return df

# ====== preprocessing ======
def process_location_col(df):
    """extract lat/lon from location string and keep only rows inside LA bounds"""

    ''' Parsed lat/long string, used ChatGPT assistance '''
    lat = df[raw_c.location].str.extract(r'\(([^,]+),')[0].astype(float)
    lon = df[raw_c.location].str.extract(r', ([^)]+)\)')[0].astype(float)

    mask = (lat >= LA_LAT_MIN) & (lat <= LA_LAT_MAX) & (lon >= LA_LON_MIN) & (lon <= LA_LON_MAX)

    df = df[mask].copy()
    df[feat_c.lat] = lat[mask]
    df[feat_c.lon] = lon[mask]

    return df

def process_datetime_col(df):
    """create combined datetime column from date and time cols"""

    ''' Formatted datetime string, used ChatGPT assistance '''
    df[feat_c.datetime] = pd.to_datetime(
        df[raw_c.date] + ' ' + df[raw_c.time].astype(str).str.zfill(4),
        format='%m/%d/%Y %H%M',
        errors='coerce'
    )
    return df

def assign_grids(df, lat_step=0.013, lon_step=0.015):
    """
    assign each row to a contiguous grid in LA (~2km^2)
    grid id is readable using bottom-left corner coordinates
    """

    df[feat_c.grid_row] = ((df[feat_c.lat] - LA_LAT_MIN) // lat_step).astype(int)
    df[feat_c.grid_col] = ((df[feat_c.lon] - LA_LON_MIN) // lon_step).astype(int)
    ''' Generated unique grid ID, used ChatGPT assistance '''
    df[feat_c.grid_id] = (
        'grid_lat' + ((df[feat_c.grid_row] * lat_step) + LA_LAT_MIN).round(3).astype(str) +
        '_lon' + ((df[feat_c.grid_col] * lon_step) + LA_LON_MIN).round(3).astype(str)
    )
    return df

def assign_week(df):
    """assign iso week number, iso year, and readable week id (monday to sunday)"""

    iso = df[feat_c.datetime].dt.isocalendar()
    df[feat_c.week_number] = iso['week']
    df[feat_c.week_year] = iso['year']
    
    ''' Extracted week start, end, and unique ID, used ChatGPT assistance '''
    df[feat_c.week_start] = pd.to_datetime(
        df[feat_c.week_year].astype(str) + '-W' + df[feat_c.week_number].astype(str) + '-1',
        format='%G-W%V-%u'
    )
    df[feat_c.week_end] = df[feat_c.week_start] + pd.Timedelta(days=6)
    df[feat_c.week_id] = df[feat_c.week_start].dt.strftime('%Y-%m-%d') + '/' + df[feat_c.week_end].dt.strftime('%Y-%m-%d')
    
    return df

# ====== augmentation ======
def aggregate_crimes_per_unit(df, top_crimes):
    """
    aggregate crime counts per spatio-temporal unit
    keep only top k most frequent crime types
    """

    # filter to top k crimes
    df = df[df[raw_c.crime_desc].isin(top_crimes)]

    # one-hot encode crime types
    crime_dummies = pd.get_dummies(df[raw_c.crime_desc])
    crime_dummies = crime_dummies.reindex(columns=top_crimes, fill_value=0)
    df = df.join(crime_dummies)

    # aggregate counts per grid_id x week_year x week_number
    group_cols = ['grid_id', 'week_year', 'week_number']
    aggregated = df.groupby(group_cols)[crime_dummies.columns].sum().reset_index()

    # sort for consistency
    aggregated = aggregated.sort_values(['grid_id', 'week_year', 'week_number'])
    
    return aggregated

def compute_rolling_avg(aggregated, top_crimes, window=2):
    """
    compute rolling average of previous 'window' weeks per grid
    """

    for crime in top_crimes:
        ''' Double-checked rolling average logic, used ChatGPT assistance '''
        aggregated[f'{crime}_rolling_{window}w'] = (
            aggregated.groupby('grid_id')[crime]
            .shift(1) # exclude current week
            .rolling(window, min_periods=1) # rolling window of previous weeks
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    return aggregated

# ====== execution ======
data = pd.read_csv(f"data/{FILE_PATH}")
data = clean(data)

data = process_location_col(data)
data = process_datetime_col(data)

data = assign_grids(data)
data = assign_week(data)

top_crimes = data[raw_c.crime_desc].value_counts().nlargest(15).index.tolist()

data = aggregate_crimes_per_unit(data, top_crimes)
data = compute_rolling_avg(data, top_crimes)
debug(data, "RESULT")