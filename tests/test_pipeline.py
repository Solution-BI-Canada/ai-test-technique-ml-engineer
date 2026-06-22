import pandas as pd

from src.preprocessing import (
    remove_leakage_features,
    split_train_test_by_date,
    fit_preprocessing_params,
    handle_missing_values,
    encode_categorical_features,
    engineer_temporal_features,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([f"2024-01-{day:02d}" for day in range(1, 11)]),
            "sku_id": [
                "SKU_001", "SKU_001", "SKU_002", "SKU_002", "SKU_001",
                "SKU_003", "SKU_003", "SKU_004", "SKU_004", "SKU_005",
            ],
            "store_id": [
                "STORE_1", "STORE_1", "STORE_2", "STORE_2", "STORE_1",
                "STORE_3", "STORE_3", "STORE_4", "STORE_4", "STORE_5",
            ],
            "sales_qty": [10.0, None, 20.0, 30.0, None, 15.0, None, 40.0, 42.0, 50.0],
            "stock_level": [50, 40, 30, 20, 10, 60, 55, 25, 20, 15],
            "promotion_flag": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "temperature": [10.0, None, 12.0, 13.0, None, 8.0, 9.0, None, 11.0, 7.0],
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "stockout_next_3d": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            "stock_risk_score": [0.01, 0.02, 0.99, 0.98, 0.01, 0.02, 0.03, 0.97, 0.98, 0.99],
        }
    )


def test_remove_leakage_features_removes_stock_risk_score():
    df = make_sample_df()

    cleaned_df = remove_leakage_features(df)

    assert "stock_risk_score" not in cleaned_df.columns
    assert "stock_risk_score" in df.columns


def test_split_train_test_by_date_returns_non_empty_temporal_sets():
    df = remove_leakage_features(make_sample_df())

    train_df, test_df = split_train_test_by_date(df)

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert train_df["date"].max() < test_df["date"].min()


def test_handle_missing_values_uses_train_params_only():
    df = remove_leakage_features(make_sample_df())
    train_df, test_df = split_train_test_by_date(df)

    params = fit_preprocessing_params(train_df)

    train_clean = handle_missing_values(train_df, params, "train")
    test_clean = handle_missing_values(test_df, params, "test")

    assert train_clean[["sales_qty", "temperature"]].isna().sum().sum() == 0
    assert test_clean[["sales_qty", "temperature"]].isna().sum().sum() == 0


def test_encode_categorical_features_handles_unknown_categories():
    df = remove_leakage_features(make_sample_df())
    train_df, test_df = split_train_test_by_date(df)

    test_df = test_df.copy()
    test_df.loc[test_df.index[0], "sku_id"] = "SKU_UNKNOWN"
    test_df.loc[test_df.index[0], "store_id"] = "STORE_UNKNOWN"

    params = fit_preprocessing_params(train_df)

    encoded_test = encode_categorical_features(test_df, params, "test")

    assert "sku_id" not in encoded_test.columns
    assert "store_id" not in encoded_test.columns

    sku_columns = [column for column in encoded_test.columns if column.startswith("sku_id_")]
    store_columns = [column for column in encoded_test.columns if column.startswith("store_id_")]

    assert sku_columns
    assert store_columns

    first_row = encoded_test.iloc[0]

    assert first_row[sku_columns].sum() == 0
    assert first_row[store_columns].sum() == 0


def test_engineer_temporal_features_removes_date_and_creates_features():
    df = remove_leakage_features(make_sample_df())
    train_df, _ = split_train_test_by_date(df)

    transformed_df = engineer_temporal_features(train_df, "train")

    assert "date" not in transformed_df.columns
    assert "year" in transformed_df.columns
    assert "month" in transformed_df.columns
    assert "day" in transformed_df.columns
    assert "day_of_year" in transformed_df.columns


def test_full_preprocessing_pipeline_train_and_test_have_valid_schema():
    df = remove_leakage_features(make_sample_df())
    train_df, test_df = split_train_test_by_date(df)

    params = fit_preprocessing_params(train_df)

    train_processed = handle_missing_values(train_df, params, "train")
    test_processed = handle_missing_values(test_df, params, "test")

    train_processed = encode_categorical_features(train_processed, params, "train")
    test_processed = encode_categorical_features(test_processed, params, "test")

    train_processed = engineer_temporal_features(train_processed, "train")
    test_processed = engineer_temporal_features(test_processed, "test")

    assert train_processed.isna().sum().sum() == 0
    assert test_processed.isna().sum().sum() == 0

    forbidden_columns = {"stock_risk_score", "sku_id", "store_id", "date"}

    assert forbidden_columns.isdisjoint(train_processed.columns)
    assert forbidden_columns.isdisjoint(test_processed.columns)

    assert list(train_processed.columns) == list(test_processed.columns)

    non_numeric_train = [
        column for column in train_processed.columns
        if not pd.api.types.is_numeric_dtype(train_processed[column])
    ]

    non_numeric_test = [
        column for column in test_processed.columns
        if not pd.api.types.is_numeric_dtype(test_processed[column])
    ]

    assert non_numeric_train == []
    assert non_numeric_test == []