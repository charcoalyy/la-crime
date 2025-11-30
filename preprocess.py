from dataclasses import dataclass
import holidays

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)

'''
AI DISCLAIMER NOTE: 
26 prompts used, where use cases are noted in the code below. Carbon usage for this file:

26*4.32g = 112.32g CO2

'''

# ====== helpers ======
def debug_df(df, msg=""):
    print(f"\nDEBUG >> {msg}")
    print(df.head(10))

# ====== constants ======
FILE_PATH = "Crime_Data_2010_2017.csv"
LA_LAT_MIN, LA_LAT_MAX = 33.0, 35.0
LA_LON_MIN, LA_LON_MAX = -119.5, -117.0

''' AI: Automated manual work of creating data class using ChatGPT assistance '''
@dataclass(frozen=True)
class raw_c:
    date: str = 'Date Occurred'
    time: str = 'Time Occurred'
    area: str = 'Area Name'
    crime_desc: str = 'Crime Code Description'
    location: str = 'Location'

''' AI: Automated manual work of creating data class using ChatGPT assistance '''
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

# ====== cleaning ======
def clean_raw(df):
    """trim spaces from col names, keep only relevant cols, drop incomplete rows"""

    df.columns = df.columns.str.strip().str.title()
    df = df[[raw_c.date, raw_c.time, raw_c.area, raw_c.crime_desc, raw_c.location]]
    df = df.dropna(subset=[raw_c.date, raw_c.time, raw_c.location, raw_c.crime_desc])
    return df

def simplify_classes(df, n_clusters=36):
    """automatically group similar crime descriptions (classes) by clustering semantic embeddings"""

    df[raw_c.crime_desc] = (
        df[raw_c.crime_desc]
        .str.replace('"', '')
        .str.lower()
        .str.strip()
    )

    crime_list = df[raw_c.crime_desc].unique().tolist()

    ''' AI: Learned how to use sentence embeddings and KMeans using ChatGPT assistance '''
    model = SentenceTransformer('all-mpnet-base-v2') # pre-trained sentence embedding model
    X = model.encode(crime_list, convert_to_numpy=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_map = {crime: label for crime, label in zip(crime_list, labels)}
    df['cluster_num'] = df[raw_c.crime_desc].map(cluster_map)

    ''' AI: Re-assign readable label by using the most frequent crime in its cluster '''
    cluster_labels = {}
    for cluster in range(n_clusters):
        crimes_in_cluster = df[df['cluster_num'] == cluster][raw_c.crime_desc]
        if not crimes_in_cluster.empty:
            most_common = crimes_in_cluster.value_counts().idxmax()
        else:
            most_common = f'cluster_{cluster}'
        cluster_labels[cluster] = most_common

    df[raw_c.crime_desc] = df['cluster_num'].map(cluster_labels)
    df = df.drop(columns=['cluster_num'])

    return df

# ====== preprocessing ======
def process_location_col(df):
    """extract lat/lon from location string and keep only rows inside LA bounds"""

    ''' AI: Parsed lat/long string using ChatGPT assistance '''
    lat = df[raw_c.location].str.extract(r'\(([^,]+),')[0].astype(float)
    lon = df[raw_c.location].str.extract(r', ([^)]+)\)')[0].astype(float)

    mask = (lat >= LA_LAT_MIN) & (lat <= LA_LAT_MAX) & (lon >= LA_LON_MIN) & (lon <= LA_LON_MAX)

    df = df[mask].copy()
    df[feat_c.lat] = lat[mask]
    df[feat_c.lon] = lon[mask]

    return df

def process_datetime_col(df):
    """create combined datetime column from date and time cols"""

    ''' AI: Formatted datetime string using ChatGPT assistance '''
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

    ''' AI: Generated unique grid ID using ChatGPT assistance '''
    df[feat_c.grid_id] = (
        'grid_lat' + ((df[feat_c.grid_row] * lat_step) + LA_LAT_MIN).round(3).astype(str) +
        '_lon' + ((df[feat_c.grid_col] * lon_step) + LA_LON_MIN).round(3).astype(str)
    )

    return df

def assign_week(df):
    """assign iso week number, iso year, start date (monday to sunday)"""

    iso = df[feat_c.datetime].dt.isocalendar()
    df[feat_c.week_number] = iso['week']
    df[feat_c.week_year] = iso['year']
    
    return df

# ====== augmentation (crime-related features) ======
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

    # aggregate counts per grid_row x grid_col x week_year x week_number
    # grid_id maintained for mapping simplicity later on
    group_cols = ['grid_id', 'grid_row', 'grid_col', 'week_year', 'week_number']
    aggregated = df.groupby(group_cols)[crime_dummies.columns].sum().reset_index()

    # sort for consistency
    aggregated = aggregated.sort_values(['grid_id', 'grid_row', 'grid_col', 'week_year', 'week_number'])
    
    return aggregated

def compute_rolling_avg(aggregated, top_crimes, window=2):
    """compute rolling average of previous 'window' weeks per grid"""

    for crime in top_crimes:
        ''' AI: Double-checked rolling average logic using ChatGPT assistance '''
        aggregated[f'{crime}_rolling_{window}w'] = (
            aggregated
            .groupby(['grid_row', 'grid_col'])[crime]
            .shift(1) # exclude current week
            .rolling(window, min_periods=1) # rolling window of previous weeks
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    return aggregated

def compute_neighbour_score(df, top_crimes, weight_inner=1.0, weight_middle=0.5, weight_outer=0.25):
    """
    compute distance-weighted neighbour crime activity score
    based on a 5x5 grid window around each grid cell.
    """

    # sum total crimes per unit
    df['total_crimes'] = df[top_crimes].sum(axis=1)

    ''' AI: Determined how to get faster lookup using ChatGPT assistance '''
    lookup = df.set_index(['week_year','week_number','grid_row','grid_col'])['total_crimes']

    ''' AI: Acquired offset of 5x5 area around center cell using ChatGPT assistance '''
    offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
               if not (dr == 0 and dc == 0)]
    
    # count neighbour crimes (weighted) to derive score
    df['neigh_activity_score'] = 0.0

    for idx, row in df.iterrows():
        wy, wn = row['week_year'], row['week_number']
        r, c = row['grid_row'], row['grid_col']

        score = 0.0

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            dist = abs(dr) + abs(dc)

            if dist == 1:
                w = weight_inner
            elif dist == 2:
                w = weight_middle
            else:
                w = weight_outer

            neigh_val = lookup.get((wy, wn, nr, nc), 0)
            score += w * neigh_val

        df.at[idx, 'neigh_activity_score'] = score

    return df.drop(columns=['grid_row','grid_col'])

# ====== augmentation (non-crime-related features) ======
def compute_season(df):
    def assign_season(row):
        ''' AI: Reconstructed week dates using ChatGPT assistance '''
        week_start = pd.to_datetime(f'{int(row.week_year)}-W{int(row.week_number):02d}-1', format='%G-W%V-%u')
        month = week_start.month

        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    df['season'] = df.apply(assign_season, axis=1)
    return df

def compute_is_holiday(df):
    us_holidays = holidays.US(years=df['week_year'].unique())

    def assign_holiday(row):
        ''' AI: Reconstructed week dates using ChatGPT assistance '''
        week_start = pd.to_datetime(f'{int(row.week_year)}-W{int(row.week_number):02d}-1', format='%G-W%V-%u')
        week_end = week_start + pd.Timedelta(days=6)

        for date in pd.date_range(week_start, week_end):
            if date in us_holidays:
                return 1
        return 0

    df['is_holiday'] = df.apply(assign_holiday, axis=1)
    return df

# ====== execution ======
data = pd.read_csv(f"data_raw/{FILE_PATH}")

data = clean_raw(data)
data = simplify_classes(data)

data = process_location_col(data)
data = process_datetime_col(data)
data = assign_grids(data)
data = assign_week(data)

top_crimes = data[raw_c.crime_desc].value_counts().nlargest(20).index.tolist()

data = aggregate_crimes_per_unit(data, top_crimes)
data = compute_rolling_avg(data, top_crimes)
data = compute_neighbour_score(data, top_crimes)

data = compute_season(data)
data = compute_is_holiday(data)

debug_df(data, "RESULT")