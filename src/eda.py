
from pathlib import Path
import sys
from typing import Final, List, Set

import pandas as pd


REQUIRED_COLUMNS: Final[Set[str]] = {
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

BUSINESS_KEY: Final[List[str]] = ["date", "sku_id", "store_id"]

NUMERIC_COLUMNS: Final[List[str]] = [
    "sales_qty",
    "stock_level",
    "temperature",
    "stock_risk_score",
]

BINARY_COLUMNS: Final[List[str]] = [
    "promotion_flag",
    "stockout_next_3d",
]

MIN_GROUP_COUNT: Final[int] = 30


def get_project_root() -> Path:
    current_path = Path.cwd().resolve()

    for candidate in [current_path] + list(current_path.parents):
        expected_file = candidate / "data" / "raw" / "stocks.csv"

        if expected_file.is_file():
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


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def validate_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Dataset is empty.")

    df = df.copy()

    original_missing_counts = df.isna().sum()

    original_dates = df["date"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_date_mask = df["date"].isna() & original_dates.notna()

    if invalid_date_mask.any():
        raise ValueError(
            f"Invalid date values found: {int(invalid_date_mask.sum())}"
        )

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["day_of_week"]:
        original_values = df[column].copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")

        invalid_conversion_mask = df[column].isna() & original_values.notna()

        if invalid_conversion_mask.any():
            raise ValueError(
                f"Invalid numeric conversion in column '{column}': "
                f"{int(invalid_conversion_mask.sum())} values"
            )

    critical_columns = BUSINESS_KEY + ["stockout_next_3d"]
    missing_critical = df[critical_columns].isna().sum()
    missing_critical = missing_critical[missing_critical > 0]

    if not missing_critical.empty:
        raise ValueError(
            f"Missing values found in critical columns: "
            f"{missing_critical.to_dict()}"
        )

    duplicate_count = int(df.duplicated(subset=BUSINESS_KEY).sum())

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate rows found on business key "
            f"date + sku_id + store_id: {duplicate_count}"
        )

    print("\n=== ORIGINAL MISSING VALUES BEFORE TYPE CONVERSION ===")
    original_missing_summary = original_missing_counts[
        original_missing_counts > 0
    ]

    if original_missing_summary.empty:
        print("No original missing values detected.")
    else:
        print(original_missing_summary)

    return df


def validate_prepared_dtypes(df: pd.DataFrame) -> None:
    expected_numeric_columns = NUMERIC_COLUMNS + BINARY_COLUMNS + ["day_of_week"]

    for column in expected_numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric after preparation.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("Column 'date' is not datetime after preparation.")


def analyze_data_structure(df: pd.DataFrame) -> None:
    print("\n=== 3.2 DATA STRUCTURE CONSISTENCY ===")

    row_count = len(df)
    unique_dates = df["date"].nunique()
    unique_skus = df["sku_id"].nunique()
    unique_stores = df["store_id"].nunique()
    unique_business_keys = df[BUSINESS_KEY].drop_duplicates().shape[0]

    print(f"Actual number of rows: {row_count}")
    print(f"Unique dates: {unique_dates}")
    print(f"Unique SKUs: {unique_skus}")
    print(f"Unique stores: {unique_stores}")
    print(f"Unique business key combinations: {unique_business_keys}")

    print("\nStructure status: PASSED BUSINESS KEY UNIQUENESS CHECK")
    print("No duplicate date + sku_id + store_id rows were found.")
    print(
        "Full dataset completeness cannot be confirmed because expected "
        "date x SKU x store coverage is DATA_NOT_AVAILABLE."
    )


def analyze_missing_values(df: pd.DataFrame) -> None:
    print("\n=== 3.3 MISSING VALUES ===")

    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percentage": missing_percentages.round(2),
        }
    )

    missing_summary = missing_summary[
        missing_summary["missing_count"] > 0
    ].sort_values(by="missing_count", ascending=False)

    if missing_summary.empty:
        print("No missing values detected.")
        return

    print(missing_summary)


def analyze_invalid_values(df: pd.DataFrame) -> None:
    print("\n=== 3.4.1 IMPOSSIBLE OR INVALID VALUES ===")

    checks = {
        "negative_sales_qty": int((df["sales_qty"].dropna() < 0).sum()),
        "negative_stock_level": int((df["stock_level"].dropna() < 0).sum()),
        "invalid_promotion_flag": int(
            (~df["promotion_flag"].dropna().isin([0, 1])).sum()
        ),
        "invalid_stockout_next_3d": int(
            (~df["stockout_next_3d"].dropna().isin([0, 1])).sum()
        ),
        "invalid_day_of_week": int(
            (~df["day_of_week"].dropna().isin(range(7))).sum()
        ),
        "temperature_bounds": "DATA_NOT_AVAILABLE",
        "stock_risk_score_bounds": "DATA_NOT_AVAILABLE",
    }

    for check_name, count in checks.items():
        print(f"{check_name}: {count}")


def analyze_outliers_iqr(df: pd.DataFrame) -> None:
    print("\n=== 3.4.2 STATISTICAL OUTLIERS - IQR METHOD ===")

    outlier_summary = []

    for column in NUMERIC_COLUMNS:
        series = df[column].dropna()

        if series.empty:
            outlier_summary.append(
                {
                    "column": column,
                    "status": "DATA_NOT_AVAILABLE",
                    "outlier_count": "DATA_NOT_AVAILABLE",
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_summary.append(
                {
                    "column": column,
                    "status": "IQR_ZERO_NOT_INFORMATIVE",
                    "outlier_count": 0,
                }
            )
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            ((series < lower_bound) | (series > upper_bound)).sum()
        )

        outlier_summary.append(
            {
                "column": column,
                "status": "computed_warning_only",
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": outlier_count,
            }
        )

    print(pd.DataFrame(outlier_summary))
    print(
        "Note: IQR outliers are warnings only. "
        "Official outlier rules are DATA_NOT_AVAILABLE."
    )


def inspect_temperature_outliers(df: pd.DataFrame) -> None:
    print("\n=== 3.4.3 TEMPERATURE OUTLIERS DETAILS ===")

    temperature = df["temperature"].dropna()

    if temperature.empty:
        print("Temperature analysis unavailable: DATA_NOT_AVAILABLE")
        return

    q1 = temperature.quantile(0.25)
    q3 = temperature.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        print("Temperature IQR is zero. Outlier analysis is not informative.")
        return

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = temperature[
        (temperature < lower_bound) | (temperature > upper_bound)
    ]

    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")

    print("\nTop 20 highest temperatures:")
    print(temperature.sort_values(ascending=False).head(20).to_list())


def analyze_target_distribution(df: pd.DataFrame) -> None:
    print("\n=== 3.5 TARGET DISTRIBUTION ===")

    target = df["stockout_next_3d"].dropna()

    if target.empty:
        raise ValueError(
            "Target distribution cannot be analyzed because "
            "stockout_next_3d is entirely missing."
        )

    target_counts = target.value_counts().sort_index()
    target_percentages = target.value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_percentages.round(2),
        }
    )

    print(summary)

    majority_class_pct = target_percentages.max()

    print(f"\nMajority class percentage: {majority_class_pct:.2f}%")

    if majority_class_pct >= 90:
        print("Dataset status: SEVERELY IMBALANCED")
    elif majority_class_pct >= 80:
        print("Dataset status: IMBALANCED")
    elif majority_class_pct >= 60:
        print("Dataset status: MODERATELY IMBALANCED")
    else:
        print("Dataset status: RELATIVELY BALANCED")


def analyze_feature_relationships(df: pd.DataFrame) -> None:
    print("\n=== 3.6.1 NUMERICAL PATTERNS AND RELATIONSHIPS ===")

    target = "stockout_next_3d"

    correlation_summary = []

    for feature in NUMERIC_COLUMNS:
        clean_pair = df[[feature, target]].dropna()

        if clean_pair.empty:
            status = "DATA_NOT_AVAILABLE"
            correlation_value = "DATA_NOT_AVAILABLE"
        elif clean_pair[feature].nunique() < 2 or clean_pair[target].nunique() < 2:
            status = "NOT_COMPUTABLE_ZERO_VARIANCE"
            correlation_value = "DATA_NOT_AVAILABLE"
        else:
            status = "computed_warning_only"
            correlation_value = round(clean_pair[feature].corr(clean_pair[target]), 4)

        correlation_summary.append(
            {
                "feature": feature,
                "status": status,
                "correlation_with_binary_target": correlation_value,
            }
        )

    print(pd.DataFrame(correlation_summary))
    print(
        "\nNote: Pearson correlation with a binary target is exploratory only."
    )

    print("\n=== TARGET GROUP COMPARISON ===")
    print(df.groupby(target)[NUMERIC_COLUMNS].mean().round(2))


def analyze_categorical_patterns(df: pd.DataFrame) -> None:
    print("\n=== 3.6.2 CATEGORICAL PATTERNS ===")

    target = "stockout_next_3d"

    categorical_features = [
        "promotion_flag",
        "day_of_week",
        "sku_id",
        "store_id",
    ]

    for feature in categorical_features:
        print(f"\n=== Target rate by {feature} ===")

        pattern_summary = (
            df.groupby(feature)[target]
            .agg(["count", "mean"])
            .rename(columns={"mean": "stockout_rate"})
        )

        pattern_summary["stockout_rate"] = (
            pattern_summary["stockout_rate"] * 100
        ).round(2)

        low_volume_summary = pattern_summary[
            pattern_summary["count"] < MIN_GROUP_COUNT
        ]

        reliable_summary = pattern_summary[
            pattern_summary["count"] >= MIN_GROUP_COUNT
        ].sort_values(by="stockout_rate", ascending=False)

        if reliable_summary.empty:
            print(
                f"No groups with at least {MIN_GROUP_COUNT} observations. "
                "Displaying raw top groups as warning only."
            )
            print(
                pattern_summary
                .sort_values(by="stockout_rate", ascending=False)
                .head(20)
            )
        else:
            print(
                f"Showing groups with at least {MIN_GROUP_COUNT} observations."
            )
            print(reliable_summary.head(20))

        if not low_volume_summary.empty:
            print(
                f"Low-volume groups excluded from ranked interpretation: "
                f"{len(low_volume_summary)}"
            )


def analyze_temporal_consistency(df: pd.DataFrame) -> None:
    print("\n=== 3.7 TEMPORAL COVERAGE ===")

    dates = df["date"].drop_duplicates().sort_values()

    print(f"First date: {dates.min().date()}")
    print(f"Last date: {dates.max().date()}")
    print(f"Number of observed dates: {len(dates)}")

    date_gaps = dates.diff().dropna().dt.days

    if date_gaps.empty:
        print("Temporal gap analysis unavailable: DATA_NOT_AVAILABLE")
        return

    print("\n=== OBSERVED DATE GAP DISTRIBUTION ===")
    print(date_gaps.value_counts().sort_index())

    print(
        "\nContinuous daily coverage is not enforced because "
        "the expected business calendar is DATA_NOT_AVAILABLE."
    )

    monthly_stockout_rate = (
        df.assign(month=df["date"].dt.to_period("M"))
        .groupby("month")["stockout_next_3d"]
        .mean()
        .mul(100)
        .round(2)
    )

    print("\n=== MONTHLY STOCKOUT RATE (%) ===")
    print(monthly_stockout_rate)


def analyze_data_leakage(df: pd.DataFrame) -> None:
    print("\n=== 3.8 DATA LEAKAGE ANALYSIS ===")

    target = "stockout_next_3d"
    leakage_feature = "stock_risk_score"

    clean_pair = df[[leakage_feature, target]].dropna()

    if clean_pair.empty:
        print("Leakage analysis unavailable: DATA_NOT_AVAILABLE")
        return

    if clean_pair[target].nunique() < 2:
        print(
            "Leakage analysis limited: target contains fewer than 2 classes. "
            "Correlation and class range comparison are not informative."
        )
        return

    if clean_pair[leakage_feature].nunique() < 2:
        print(
            "Leakage analysis limited: stock_risk_score has zero variance. "
            "Correlation is not informative."
        )
        return

    correlation = clean_pair[leakage_feature].corr(clean_pair[target])

    print(
        f"Correlation between {leakage_feature} and {target}: "
        f"{correlation:.4f}"
    )

    grouped = clean_pair.groupby(target)[leakage_feature].describe().round(4)

    print("\n=== STOCK_RISK_SCORE BY TARGET ===")
    print(grouped)

    target_zero = clean_pair.loc[clean_pair[target] == 0, leakage_feature]
    target_one = clean_pair.loc[clean_pair[target] == 1, leakage_feature]

    print("\n=== RANGE COMPARISON ===")
    print(f"Class 0 range: {target_zero.min():.4f} -> {target_zero.max():.4f}")
    print(f"Class 1 range: {target_one.min():.4f} -> {target_one.max():.4f}")


    if abs(correlation) >= 0.95:
        print(
            "\nWARNING: Very high correlation detected. "
            "The feature appears to be almost perfectly correlated "
            "with the target. This strongly suggests potential "
            "data leakage and requires investigation before model training."
        )
    else:
        print(
            "\nNo strong evidence of data leakage was detected "
            "based on correlation analysis alone."
        )


def main() -> None:
    try:
        project_root = get_project_root()
        file_path = project_root / "data" / "raw" / "stocks.csv"

        df = load_stock_data(file_path)
        validate_schema(df)

        df = validate_and_prepare_data(df)
        validate_prepared_dtypes(df)

        analyze_data_structure(df)
        analyze_missing_values(df)

        analyze_invalid_values(df)
        analyze_outliers_iqr(df)
        inspect_temperature_outliers(df)

        analyze_target_distribution(df)

        analyze_feature_relationships(df)
        analyze_categorical_patterns(df)

        analyze_temporal_consistency(df)
        analyze_data_leakage(df)

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
        pd.errors.ParserError,
    ) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()