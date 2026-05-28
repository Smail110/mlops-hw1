import numpy as np
import pandas as pd

from src.config import CAT_FEATURES, SELECTED_FEATURES


RAW_DEFAULTS = {
    "transaction_time": "",
    "merch": "unknown",
    "cat_id": "unknown",
    "amount": 0.0,
    "name_1": "unknown",
    "name_2": "unknown",
    "gender": "unknown",
    "street": "unknown",
    "one_city": "unknown",
    "us_state": "unknown",
    "post_code": "unknown",
    "lat": 0.0,
    "lon": 0.0,
    "population_city": 0.0,
    "jobs": "unknown",
    "merchant_lat": 0.0,
    "merchant_lon": 0.0,
}


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0

    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce").fillna(0.0))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce").fillna(0.0))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce").fillna(0.0))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce").fillna(0.0))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def count_comma_values(value) -> int:
    if pd.isna(value):
        return 0
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return len(parts)


def _ensure_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, default in RAW_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_raw_columns(df)

    for col in ["amount", "lat", "lon", "population_city", "merchant_lat", "merchant_lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    transaction_time = pd.to_datetime(df["transaction_time"], errors="coerce")
    fallback_time = pd.Timestamp("2019-01-01 00:00:00")
    transaction_time = transaction_time.fillna(fallback_time)

    df["transaction_year"] = transaction_time.dt.year
    df["transaction_month"] = transaction_time.dt.month
    df["transaction_hour"] = transaction_time.dt.hour
    df["transaction_minute"] = transaction_time.dt.minute
    df["transaction_dayofweek"] = transaction_time.dt.dayofweek
    df["transaction_dayofyear"] = transaction_time.dt.dayofyear
    df["transaction_weekofyear"] = (
        transaction_time.dt.isocalendar().week.astype("float")
    )

    df["transaction_is_weekend"] = (
        df["transaction_dayofweek"].isin([5, 6]).astype(int)
    )
    df["transaction_is_night"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["transaction_is_morning"] = df["transaction_hour"].between(6, 11).astype(int)
    df["transaction_is_day"] = df["transaction_hour"].between(12, 17).astype(int)
    df["transaction_is_evening"] = df["transaction_hour"].between(18, 23).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["transaction_dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["transaction_month"] / 12)

    df["log_amount"] = np.log1p(np.clip(df["amount"], a_min=0, a_max=None))

    df["street"] = df["street"].fillna("unknown").astype(str).str.strip()
    street_number = df["street"].str.extract(r"^\s*(\d+)")[0]
    df["street_has_start_number"] = street_number.notna().astype(int)

    df["client_geo_sector"] = (
        df["lat"].round(1).astype(str) + "_" + df["lon"].round(1).astype(str)
    )
    df["merchant_geo_sector"] = (
        df["merchant_lat"].round(1).astype(str)
        + "_"
        + df["merchant_lon"].round(1).astype(str)
    )
    df["same_geo_sector"] = (
        df["client_geo_sector"] == df["merchant_geo_sector"]
    ).astype(int)

    df["distance_to_merchant_km"] = haversine(
        df["lat"],
        df["lon"],
        df["merchant_lat"],
        df["merchant_lon"],
    )
    df["log_distance_to_merchant"] = np.log1p(df["distance_to_merchant_km"])

    df["jobs"] = df["jobs"].fillna("unknown").astype(str).str.strip()
    df["jobs_count"] = df["jobs"].apply(count_comma_values)

    for col in CAT_FEATURES:
        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )

    return df[SELECTED_FEATURES].copy()
