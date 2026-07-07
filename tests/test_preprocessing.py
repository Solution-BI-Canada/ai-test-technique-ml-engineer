import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocessing_creates_expected_features():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "sku_id": ["SKU_001"],
            "store_id": ["STORE_1"],
            "sales_qty": [25.0],
            "stock_level": [10],
            "promotion_flag": [0],
            "temperature": [5.0],
            "day_of_week": [0],
            "stockout_next_3d": [1],
            "stock_risk_score": [0.99],
        }
    )

    processed = preprocess_data(df)

    assert "sales_stock_gap" in processed.columns
    assert "days_of_stock" in processed.columns
    assert "stock_risk_score" not in processed.columns
    assert processed.loc[0, "sales_stock_gap"] == 15.0


def test_preprocessing_handles_invalid_temperature_and_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "sku_id": ["SKU_001", "SKU_001"],
            "store_id": ["STORE_1", "STORE_1"],
            "sales_qty": [np.nan, 20.0],
            "stock_level": [10, 30],
            "promotion_flag": [0, 1],
            "temperature": [65.0, 5.0],
            "day_of_week": [0, 1],
            "stockout_next_3d": [1, 0],
            "stock_risk_score": [0.99, 0.01],
        }
    )

    processed = preprocess_data(df)

    assert processed["sales_qty"].isna().sum() == 0
    assert processed["temperature"].isna().sum() == 0
    assert processed["days_of_stock"].replace([np.inf, -np.inf], np.nan).notna().all()