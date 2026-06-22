import pandas as pd

from src.preprocessing import (
    remove_leakage_features,
    split_train_test_by_date,
    fit_preprocessing_params,
    handle_missing_values,
    encode_categorical_features,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                ]
            ),
            "sku_id": [
                "SKU_001",
                "SKU_001",
                "SKU_002",
                "SKU_002",
                "SKU_001",
                "SKU_003",
                "SKU_003",
                "SKU_004",
                "SKU_004",
                "SKU_005",
            ],
            "store_id": [
                "STORE_1",
                "STORE_1",
                "STORE_2",
                "STORE_2",
                "STORE_1",
                "STORE_3",
                "STORE_3",
                "STORE_4",
                "STORE_4",
                "STORE_5",
            ],
            "sales_qty": [
                10.0,
                None,
                20.0,
                30.0,
                None,
                15.0,
                None,
                40.0,
                42.0,
                50.0,
            ],
            "stock_level": [50, 40, 30, 20, 10, 60, 55, 25, 20, 15],
            "promotion_flag": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "temperature": [10.0, None, 12.0, 13.0, None, 8.0, 9.0, None, 11.0, 7.0],
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "stockout_next_3d": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            "stock_risk_score": [
                0.01,
                0.02,
                0.99,
                0.98,
                0.01,
                0.02,
                0.03,
                0.97,
                0.98,
                0.99,
            ],
        }
    )


def test_remove_leakage_features() -> None:
    df = make_sample_df()

    result = remove_leakage_features(df)

    assert "stock_risk_score" not in result.columns
    assert "stockout_next_3d" in result.columns


def test_temporal_split_train_before_test() -> None:
    df = remove_leakage_features(make_sample_df())

    train_df, test_df = split_train_test_by_date(df)

    assert train_df["date"].max() < test_df["date"].min()
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_missing_values_are_imputed() -> None:
    df = remove_leakage_features(make_sample_df())
    train_df, _ = split_train_test_by_date(df)

    params = fit_preprocessing_params(train_df)
    result = handle_missing_values(train_df, params, "train")

    assert result["sales_qty"].isna().sum() == 0
    assert result["temperature"].isna().sum() == 0


def test_one_hot_encoding_removes_categorical_columns() -> None:
    df = remove_leakage_features(make_sample_df())
    train_df, _ = split_train_test_by_date(df)

    params = fit_preprocessing_params(train_df)
    clean_df = handle_missing_values(train_df, params, "train")
    encoded_df = encode_categorical_features(clean_df, params, "train")

    assert "sku_id" not in encoded_df.columns
    assert "store_id" not in encoded_df.columns
    assert any(column.startswith("sku_id_") for column in encoded_df.columns)
    assert any(column.startswith("store_id_") for column in encoded_df.columns)