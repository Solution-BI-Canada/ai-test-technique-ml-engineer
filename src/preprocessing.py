from pathlib import Path
import json
import sys
from typing import Any, Final

import pandas as pd


LEAKAGE_COLUMNS: Final[list[str]] = ["stock_risk_score"]
TARGET_COLUMN: Final[str] = "stockout_next_3d"
DATE_COLUMN: Final[str] = "date"

RAW_REQUIRED_COLUMNS: Final[set[str]] = {
    "date",
    "sku_id",
    "store_id",
    "sales_qty",
    "stock_level",
    "promotion_flag",
    "temperature",
    "day_of_week",
    "stockout_next_3d",
    "stock_risk_score",
}

OUTPUT_REQUIRED_COLUMNS: Final[set[str]] = RAW_REQUIRED_COLUMNS - set(LEAKAGE_COLUMNS)

NUMERIC_COLUMNS: Final[list[str]] = [
    "sales_qty",
    "stock_level",
    "temperature",
]

BINARY_COLUMNS: Final[list[str]] = [
    "promotion_flag",
    "stockout_next_3d",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "sku_id",
    "store_id",
]

TRAIN_RATIO: Final[float] = 0.8
MAX_CATEGORY_COUNT: Final[int] = 500


def get_project_root() -> Path:
    current_path = Path.cwd().resolve()

    for candidate in [current_path] + list(current_path.parents):
        expected_path = candidate / "data" / "raw" / "stocks.csv"

        if expected_path.is_file():
            return candidate

    raise FileNotFoundError(
        "Project root could not be detected. "
        "Expected file: data/raw/stocks.csv"
    )


def load_stock_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV file could not be parsed: {file_path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV file encoding could not be decoded: {file_path}") from exc


def validate_schema(df: pd.DataFrame, required_columns: set[str], name: str) -> None:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} missing required columns: {sorted(missing_columns)}"
        )


def validate_and_prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df, RAW_REQUIRED_COLUMNS, "Raw input dataframe")

    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Invalid or missing dates detected.")

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["day_of_week"]:
        original_values = df[column].copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")

        invalid_mask = df[column].isna() & original_values.notna()

        if invalid_mask.any():
            raise ValueError(
                f"Invalid numeric values found in '{column}': "
                f"{int(invalid_mask.sum())}"
            )

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)

    critical_columns = [DATE_COLUMN, "sku_id", "store_id", TARGET_COLUMN]

    if df[critical_columns].isna().any().any():
        raise ValueError("Missing values found in critical columns.")

    if df.duplicated(subset=[DATE_COLUMN, "sku_id", "store_id"]).any():
        raise ValueError(
            "Duplicate rows found on business key date + sku_id + store_id."
        )

    if (df["sales_qty"].dropna() < 0).any():
        raise ValueError("Negative sales_qty detected.")

    if (df["stock_level"].dropna() < 0).any():
        raise ValueError("Negative stock_level detected.")

    for column in BINARY_COLUMNS:
        if not set(df[column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Invalid binary values detected in {column}.")

    if not set(df["day_of_week"].dropna().unique()).issubset(set(range(7))):
        raise ValueError("Invalid day_of_week values detected.")

    return df


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 4.1 REMOVE LEAKAGE FEATURES ===")

    missing_leakage_columns = [
        column for column in LEAKAGE_COLUMNS if column not in df.columns
    ]

    if missing_leakage_columns:
        raise ValueError(
            f"Expected leakage columns are missing before removal: "
            f"{missing_leakage_columns}"
        )

    cleaned_df = df.drop(columns=LEAKAGE_COLUMNS)

    if any(column in cleaned_df.columns for column in LEAKAGE_COLUMNS):
        raise ValueError("Leakage removal failed.")

    print(f"Columns before removal: {len(df.columns)}")
    print(f"Columns after removal: {len(cleaned_df.columns)}")
    print("Leakage removal status: PASSED")

    return cleaned_df


def split_train_test_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== 4.5 TEMPORAL TRAIN / TEST SPLIT ===")

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    unique_dates = df[DATE_COLUMN].drop_duplicates().sort_values()

    split_date_index = int(len(unique_dates) * TRAIN_RATIO)

    if split_date_index <= 0 or split_date_index >= len(unique_dates):
        raise ValueError(
            "Train/test split failed because there are not enough unique dates."
        )

    train_dates = set(unique_dates.iloc[:split_date_index])
    test_dates = set(unique_dates.iloc[split_date_index:])

    train_df = df[df[DATE_COLUMN].isin(train_dates)].copy()
    test_df = df[df[DATE_COLUMN].isin(test_dates)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty dataset.")

    if train_df[DATE_COLUMN].max() >= test_df[DATE_COLUMN].min():
        raise ValueError("Temporal split failed: train dates overlap test dates.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train max date: {train_df[DATE_COLUMN].max().date()}")
    print(f"Test min date: {test_df[DATE_COLUMN].min().date()}")

    return train_df, test_df


def fit_preprocessing_params(train_df: pd.DataFrame) -> dict[str, Any]:
    print("\n=== FIT PREPROCESSING PARAMS ON TRAIN ONLY ===")

    sku_sales_medians = (
        train_df.groupby("sku_id")["sales_qty"]
        .median()
        .dropna()
        .to_dict()
    )

    global_sales_median = train_df["sales_qty"].median()
    temperature_median = train_df["temperature"].median()

    if pd.isna(global_sales_median):
        raise ValueError("Cannot calculate train global sales_qty median.")

    if pd.isna(temperature_median):
        raise ValueError("Cannot calculate train temperature median.")

    categories: dict[str, list[str]] = {}

    for column in CATEGORICAL_COLUMNS:
        learned_categories = sorted(
            train_df[column].dropna().astype(str).unique().tolist()
        )

        if len(learned_categories) > MAX_CATEGORY_COUNT:
            raise ValueError(
                f"Column {column} has too many categories for one-hot encoding: "
                f"{len(learned_categories)}"
            )

        categories[column] = learned_categories

    params: dict[str, Any] = {
        "sku_sales_medians": sku_sales_medians,
        "global_sales_median": float(global_sales_median),
        "temperature_median": float(temperature_median),
        "categories": categories,
        "unknown_category_strategy": "all_zero",
        "max_category_count": MAX_CATEGORY_COUNT,
    }

    print("Preprocessing parameters fitted on train only.")
    return params


def handle_missing_values(
    df: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> pd.DataFrame:
    print(f"\n=== 4.2 HANDLE MISSING VALUES: {name} ===")

    cleaned_df = df.copy()

    missing_before = cleaned_df[["sales_qty", "temperature"]].isna().sum()
    print("Missing values before imputation:")
    print(missing_before)

    sku_medians = cleaned_df["sku_id"].map(params["sku_sales_medians"])

    cleaned_df["sales_qty"] = cleaned_df["sales_qty"].fillna(sku_medians)
    cleaned_df["sales_qty"] = cleaned_df["sales_qty"].fillna(
        params["global_sales_median"]
    )

    cleaned_df["temperature"] = cleaned_df["temperature"].fillna(
        params["temperature_median"]
    )

    missing_after = cleaned_df[["sales_qty", "temperature"]].isna().sum()
    print("\nMissing values after imputation:")
    print(missing_after)

    if missing_after.sum() != 0:
        raise ValueError(
            f"{name}: missing value treatment failed: "
            f"{missing_after.to_dict()}"
        )

    print(f"{name} missing value treatment status: PASSED")

    return cleaned_df


def encode_categorical_features(
    df: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> pd.DataFrame:
    print(f"\n=== 4.3 ENCODE CATEGORICAL FEATURES: {name} ===")

    encoded_df = df.copy()
    columns_before = len(encoded_df.columns)

    for column in CATEGORICAL_COLUMNS:
        known_categories = params["categories"][column]
        encoded_df[column] = encoded_df[column].astype(str)

        unknown_count = int((~encoded_df[column].isin(known_categories)).sum())

        if unknown_count > 0:
            print(
                f"Warning: {unknown_count} unknown values in {column}. "
                "They will be encoded as all-zero."
            )

        for category in known_categories:
            encoded_column = f"{column}_{category}"
            encoded_df[encoded_column] = (
                encoded_df[column] == category
            ).astype(int)

        encoded_df = encoded_df.drop(columns=[column])

        print(f"{column}: {len(known_categories)} one-hot columns created")

    remaining_categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in encoded_df.columns
    ]

    if remaining_categorical_columns:
        raise ValueError(
            f"{name}: categorical columns still present after encoding: "
            f"{remaining_categorical_columns}"
        )

    print(f"Columns before encoding: {columns_before}")
    print(f"Columns after encoding: {len(encoded_df.columns)}")
    print(f"{name} categorical encoding status: PASSED")

    return encoded_df


def engineer_temporal_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"\n=== 4.4 TEMPORAL FEATURE ENGINEERING: {name} ===")

    engineered_df = df.copy()

    if DATE_COLUMN not in engineered_df.columns:
        raise ValueError(f"{name}: date column missing before temporal features.")

    engineered_df["year"] = engineered_df[DATE_COLUMN].dt.year
    engineered_df["month"] = engineered_df[DATE_COLUMN].dt.month
    engineered_df["day"] = engineered_df[DATE_COLUMN].dt.day
    engineered_df["day_of_year"] = engineered_df[DATE_COLUMN].dt.dayofyear

    engineered_df = engineered_df.drop(columns=[DATE_COLUMN])

    print("Created temporal features: ['year', 'month', 'day', 'day_of_year']")
    print("Dropped original date column")
    print(f"{name} temporal feature engineering status: PASSED")

    return engineered_df


def apply_preprocessing(
    df: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> pd.DataFrame:
    processed_df = handle_missing_values(df, params, name)
    processed_df = encode_categorical_features(processed_df, params, name)
    processed_df = engineer_temporal_features(processed_df, name)

    return processed_df


def validate_final_dataset(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== 4.6 FINAL VALIDATION: {name} ===")

    forbidden_columns = LEAKAGE_COLUMNS + CATEGORICAL_COLUMNS + [DATE_COLUMN]
    remaining_forbidden_columns = [
        column for column in forbidden_columns if column in df.columns
    ]

    if remaining_forbidden_columns:
        raise ValueError(
            f"{name}: forbidden columns still present: "
            f"{remaining_forbidden_columns}"
        )

    if df.isna().any().any():
        raise ValueError(f"{name}: missing values still present.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{name}: target column missing.")

    if not set(df[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{name}: invalid target values.")

    non_numeric_columns = [
        column for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"{name}: non-numeric columns found after preprocessing: "
            f"{non_numeric_columns}"
        )

    print(f"{name} validation status: PASSED")
    print(f"{name} shape: {df.shape}")


def save_json(
    params: dict[str, Any],
    output_path: Path,
    overwrite: bool = True,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(params, file, indent=2)


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path,
    overwrite: bool = True,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    reloaded_df = pd.read_csv(output_path)

    if len(reloaded_df) != len(df):
        raise ValueError("Saved file validation failed: row count mismatch.")

    if list(reloaded_df.columns) != list(df.columns):
        raise ValueError("Saved file validation failed: column order mismatch.")

    non_numeric_columns = [
        column for column in reloaded_df.columns
        if not pd.api.types.is_numeric_dtype(reloaded_df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"Saved file validation failed: non-numeric columns found: "
            f"{non_numeric_columns}"
        )


def main() -> None:
    try:
        project_root = get_project_root()

        input_path = project_root / "data" / "raw" / "stocks.csv"

        train_output_path = (
            project_root / "data" / "processed" / "train_preprocessed.csv"
        )
        test_output_path = (
            project_root / "data" / "processed" / "test_preprocessed.csv"
        )
        params_output_path = (
            project_root / "data" / "processed" / "preprocessing_params.json"
        )

        df = load_stock_data(input_path)
        df = validate_and_prepare_raw_data(df)

        df = remove_leakage_features(df)
        validate_schema(df, OUTPUT_REQUIRED_COLUMNS, "Post-leakage dataframe")

        train_df, test_df = split_train_test_by_date(df)

        params = fit_preprocessing_params(train_df)

        train_processed = apply_preprocessing(train_df, params, "train")
        test_processed = apply_preprocessing(test_df, params, "test")

        validate_final_dataset(train_processed, "train")
        validate_final_dataset(test_processed, "test")

        if list(train_processed.columns) != list(test_processed.columns):
            raise ValueError("Train and test column schemas do not match.")

        save_processed_data(train_processed, train_output_path, overwrite=True)
        save_processed_data(test_processed, test_output_path, overwrite=True)
        save_json(params, params_output_path, overwrite=True)

        print("\nPreprocessing status: PASSED")
        print(f"Train output: {train_output_path}")
        print(f"Test output: {test_output_path}")
        print(f"Params output: {params_output_path}")

    except (
        FileNotFoundError,
        FileExistsError,
        PermissionError,
        ValueError,
        TypeError,
        pd.errors.ParserError,
    ) as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()