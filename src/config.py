from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "stocks.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "stockout_model.joblib"

TARGET_COLUMN = "stockout_next_3d"

NUMERIC_FEATURES = [
    "sales_qty",
    "stock_level",
    "promotion_flag",
    "temperature",
    "day_of_week",
    "sales_stock_gap",
    "days_of_stock",
    "demand_exceeds_stock",
]


CATEGORICAL_FEATURES = [
    "sku_id",
    "store_id",
]

DROPPED_COLUMNS = [
    "stock_risk_score", 
]

RANDOM_STATE = 42
TEST_SIZE_DAYS = 60