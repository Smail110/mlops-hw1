from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "input" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fraud_hash_logreg.json"

TARGET_COL = "target"
INDEX_COL = "index"
PREDICTION_COL = "prediction"

SELECTED_FEATURES = [
    "merch",
    "cat_id",
    "amount",
    "name_2",
    "gender",
    "street",
    "one_city",
    "us_state",
    "lat",
    "lon",
    "population_city",
    "jobs",
    "merchant_lat",
    "merchant_lon",
    "transaction_year",
    "transaction_month",
    "transaction_hour",
    "transaction_minute",
    "transaction_dayofweek",
    "transaction_dayofyear",
    "transaction_weekofyear",
    "transaction_is_weekend",
    "transaction_is_night",
    "transaction_is_morning",
    "transaction_is_day",
    "transaction_is_evening",
    "hour_sin",
    "hour_cos",
    "dayofweek_cos",
    "month_sin",
    "log_amount",
    "street_has_start_number",
    "client_geo_sector",
    "merchant_geo_sector",
    "same_geo_sector",
    "distance_to_merchant_km",
    "log_distance_to_merchant",
    "jobs_count",
]

CAT_FEATURES = [
    "merch",
    "cat_id",
    "name_2",
    "gender",
    "street",
    "one_city",
    "us_state",
    "jobs",
    "client_geo_sector",
    "merchant_geo_sector",
]

NUMERIC_FEATURES = [
    feature for feature in SELECTED_FEATURES if feature not in CAT_FEATURES
]
