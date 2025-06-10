import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from io import StringIO
from fastapi import UploadFile


async def generate_weather_data(start_date, end_date):

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 37.4241,
        "longitude": -122.1661,
        "hourly": ["temperature_2m", "wind_speed_10m", "precipitation"],
        # "timezone": "none",
        "start_date": start_date,
        "end_date": end_date
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"timestamp": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m (°C)"] = hourly_temperature_2m
    hourly_data["wind_speed_10m (km/h)"] = hourly_wind_speed_10m
    hourly_data["precipitation (mm)"] = hourly_precipitation

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    return hourly_dataframe

async def process_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if 'Hour' not in df.columns or 'PEDOCS Score' not in df.columns:
        raise ValueError("CSV must include 'Hour' and 'PEDOCS Score' columns.")

    df = df[['Hour', 'PEDOCS Score']].copy()
    df.rename(columns={'Hour': 'timestamp', 'PEDOCS Score': 'pedocs_score'}, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
    df['pedocs_score'] = pd.to_numeric(df['pedocs_score'], errors='coerce')

    df.dropna(subset=['timestamp', 'pedocs_score'], inplace=True)
    df = df.sort_values('timestamp')

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # Day of week (0 = Monday, 6 = Sunday)
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek

    # Hour of day (0-23)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

    # Is weekend (Saturday = 5, Sunday = 6)
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0).astype(int)

    df['is_peak_hour'] = ((df['hour'] == 0) | (df['hour'] > 18)).astype(int)
    df['is_valley_hour'] = ((df['hour'] >= 5) & (df['hour'] <= 9)).astype(int)

    # Create rolling means for pedocs_score at different time windows
    df['pedocs_score_rolling_3h'] = df['pedocs_score'].rolling(window=3, min_periods=1).mean()
    df['pedocs_score_rolling_6h'] = df['pedocs_score'].rolling(window=6, min_periods=1).mean()
    df['pedocs_score_rolling_12h'] = df['pedocs_score'].rolling(window=12, min_periods=1).mean()
    df['pedocs_score_rolling_24h'] = df['pedocs_score'].rolling(window=24, min_periods=1).mean()
    df['pedocs_score_rolling_48h'] = df['pedocs_score'].rolling(window=48, min_periods=1).mean()

    # Trends over hours
    # df['trend_pedocs_score_6h'] = df['pedocs_score'].diff(periods=6)

    return df
